#!/usr/bin/env python3
"""CPU-only tests for rocm_tools/rdna2/prepare_mtp_source.py.

Everything runs against a REAL local HTTP server (http.server on
127.0.0.1) serving a synthetic mini "HF repo" whose safetensors shards are
built byte-by-byte in this file: no network, no GPU, no torch, no mocking of
the code under test -- the tool's own urllib stack talks to the fixture, and
misbehaving-server modes (ignored Range, wrong Content-Range, truncated
bodies, early socket close, 302 redirects, 404s) are produced by the handler.

The properties proven here are the ones whose failure would corrupt or
contaminate the 3-bit-vs-5-bit comparison source:
  * byte-exact roundtrip of the selected mtp.* payloads through
    Range-fetch -> cache -> repackage, original names preserved, and
    trunk/decoy keys (including "mtp." substrings inside trunk keys) never
    pulled;
  * fail-closed rejection of Range-ignoring or lying servers: no silent
    full-shard fallback, no partial acceptance, no output on failure;
  * corrupt-header rejection: overlap, out-of-range, duplicate keys,
    dtype/shape/span mismatch, oversized headers, zero-byte and non-float
    selections abort BEFORE any payload bytes are requested;
  * pinned identity: non-40-hex revisions are refused with zero requests;
    a completed manifest makes reruns zero-network; tampered output, cache
    or saved config abort instead of silently re-deriving provenance;
  * provenance manifest contents (repo/revision, full shard header JSON +
    hashes, per-tensor dtype/shape/offsets/SHA256, config/index/output
    hashes) and deterministic repackaging: identical output bytes across
    sequential and concurrent (jobs=4) runs, and a resumed in-progress run
    rebuilds the byte-identical file from proven cache alone.

Run from the repo root (or anywhere):
    python3 -m pytest -q rocm_tools/rdna2/tests/test_prepare_mtp_source_cpu.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import struct
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rocm_tools.rdna2 import prepare_mtp_source as pms   # stdlib-only module

REPO_ID = "Qwen/Qwen3.8-Flash-Next"
REV = "0123456789abcdef0123456789abcdef01234567"          # synthetic pinned SHA
GOOD_TENSORS = {
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.fc.weight",
    "mtp.norm.weight",
}
DECOYS = {
    "blocks.5.mtp.old.weight",          # "mtp." substring inside a trunk key
    "model.mtp.extra.weight",           # mtp. not at the key start
    "mtpx.norm.weight",                 # starts with "mtp" but not "mtp."
}
SHARD_NAMES = ["model-00001.safetensors", "model-00002.safetensors",
               "model-00003.safetensors"]


def det_bytes(seed: str, n: int) -> bytes:
    out = b""
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"{seed}:{i}".encode()).digest()
        i += 1
    return out[:n]


def st_build(entries, metadata=None) -> bytes:
    """Build real safetensors bytes: entries = [(name, dtype, shape, data)]."""
    header, off, blobs = {}, 0, []
    for name, dtype, shape, data in entries:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    if metadata is not None:
        header["__metadata__"] = metadata
    body = json.dumps(header).encode("utf-8")
    body += b" " * ((-len(body)) % 8)
    return struct.pack("<Q", len(body)) + body + b"".join(blobs)


def make_repo(extra=None):
    """Synthetic pinned repo: (files, payloads, weight_map).
    extra: [(shard_idx, name, dtype, shape, nbytes)] injected tensors."""
    s1 = [("model.layers.0.self_attn.q_proj.weight", "BF16", [32], det_bytes("trunk1", 64)),
          ("mtp.layers.0.self_attn.q_proj.weight", "BF16", [100], det_bytes("mtpq", 200)),
          ("blocks.5.mtp.old.weight", "BF16", [16], det_bytes("decoy1", 32))]
    s2 = [("mtp.layers.0.mlp.gate_proj.weight", "F32", [50], det_bytes("mtpg", 200)),
          ("model.mtp.extra.weight", "BF16", [8], det_bytes("decoy2", 16)),
          ("mtpx.norm.weight", "BF16", [8], det_bytes("decoy3", 16)),
          ("model.layers.0.mlp.down_proj.weight", "BF16", [16], det_bytes("trunk2", 32))]
    s3 = [("mtp.fc.weight", "BF16", [64], det_bytes("mtpfc", 128)),
          ("mtp.norm.weight", "BF16", [8], det_bytes("mtpnorm", 16))]
    shards = [s1, s2, s3]
    for (idx, name, dtype, shape, nbytes) in (extra or []):
        shards[idx].append((name, dtype, shape, det_bytes(f"special:{name}", nbytes)))
    names = dict(zip(SHARD_NAMES, shards))
    files, weight_map = {}, {}
    for fname, entries in names.items():
        meta = {"format": "pt"} if fname == "model-00003.safetensors" else None
        files[fname] = st_build(entries, metadata=meta)
        for name, _dtype, _shape, _data in entries:
            weight_map[name] = fname
    index = {"metadata": {"total_size": sum(len(v) for v in files.values())},
             "weight_map": weight_map}
    config = {"architectures": ["Qwen38FlashNextForCausalLM"],
              "model_type": "qwen38_flash_next", "hidden_size": 8}
    files[pms.DEFAULT_INDEX_FILE] = json.dumps(index, indent=1).encode()
    files[pms.DEFAULT_CONFIG_FILE] = json.dumps(config, indent=1).encode()
    payloads = {name: data for grp in shards for (name, _d, _s, data) in grp}
    return files, payloads, weight_map


class FakeHF:
    """Local pinned-repo server with server-behaviour modes.
    Log entries: (path, range_header_or_None, status, served_bytes)."""

    PATH_RE = re.compile(r"^/Qwen/Qwen3\.8-Flash-Next/resolve/([0-9a-f]{40})/(.+)$")

    def __init__(self, files, mode="normal"):
        self.files = files
        self.mode = mode
        self.log = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.handle(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        return False

    def _record(self, path, rng, status, nbytes):
        with self._lock:
            self.log.append((path, rng, status, nbytes))

    def _send(self, h, status, body, headers=None):
        h.send_response(status)
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        if body:
            h.wfile.write(body)

    def handle(self, h):
        parsed = urllib.parse.urlsplit(h.path)
        rng = h.headers.get("Range")
        m = self.PATH_RE.match(urllib.parse.unquote(parsed.path))
        if not m or m.group(1) != REV:
            self._record(h.path, rng, 404, 0)
            return self._send(h, 404, b"no such pinned path")
        name = m.group(2)
        if self.mode == "not_found_shard2" and name == "model-00002.safetensors":
            self._record(h.path, rng, 404, 0)
            return self._send(h, 404, b"missing shard")
        if self.mode == "redirect" and not parsed.query:
            self._record(h.path, rng, 302, 0)
            h.send_response(302)
            h.send_header("Location", h.path + "?cachebust=1")
            h.send_header("Content-Length", "0")
            h.end_headers()
            return
        data = self.files.get(name)
        if data is None:
            self._record(h.path, rng, 404, 0)
            return self._send(h, 404, b"missing")
        if rng is None or self.mode == "ignore_range":
            # ignore_range: even Range requests get a 200 + whole file --
            # exactly what the tool must refuse (no full-download fallback).
            self._record(h.path, rng, 200, len(data))
            return self._send(h, 200, data,
                              {"Content-Type": "application/octet-stream"})
        gm = re.match(r"^bytes=(\d+)-(\d+)$", rng)
        if not gm:
            self._record(h.path, rng, 400, 0)
            return self._send(h, 400, b"bad range")
        a, b = int(gm.group(1)), int(gm.group(2))
        if a >= len(data):
            self._record(h.path, rng, 416, 0)
            return self._send(h, 416, b"")
        b = min(b, len(data) - 1)
        span = data[a:b + 1]
        cr = f"bytes {a}-{b}/{len(data)}"
        if self.mode == "bad_content_range":
            cr = f"bytes {a + 1}-{b + 1}/{len(data)}"   # lies about the span
        self._record(h.path, rng, 206, len(span))
        h.send_response(206)
        h.send_header("Content-Range", cr)
        h.send_header("Accept-Ranges", "bytes")
        if self.mode == "truncate_cl":
            half = span[: max(1, len(span) // 2)]
            h.send_header("Content-Length", str(len(half)))
            h.end_headers()
            h.wfile.write(half)
            return
        if self.mode == "early_close":
            half = span[: max(1, len(span) // 2)]
            h.send_header("Content-Length", str(len(span)))   # advertises, then dies
            h.end_headers()
            try:
                h.wfile.write(half)
                h.wfile.flush()
            except OSError:
                pass
            h.close_connection = True
            return
        if self.mode == "flaky_once":
            with self._lock:
                n_206 = sum(1 for e in self.log if e[2] == 206)
            if n_206 == 0:                    # first ranged response is corrupt
                half = span[: max(1, len(span) // 2)]
                self._record(h.path, rng, 206, 0)
                h.send_response(206)
                h.send_header("Content-Range", cr)
                h.send_header("Content-Length", str(len(half)))
                h.end_headers()
                h.wfile.write(half)
                return
        h.send_header("Content-Length", str(len(span)))
        h.end_headers()
        h.wfile.write(span)


def run_tool(base_url, out_dir=None, extra=()):
    argv = ["--repo", REPO_ID, "--revision", REV, "--base-url", base_url,
            "--retries", "2", "--backoff-max", "0", "--timeout", "5", "--quiet"]
    if out_dir is not None:
        argv += ["--output-dir", str(out_dir)]
    argv += list(extra)
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        rc = pms.main(argv)
    return rc, so.getvalue(), se.getvalue()


def read_st(path: Path):
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    payload = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        b, e = v["data_offsets"]
        payload[k] = raw[8 + n + b: 8 + n + e]
    return n, header, payload, raw


class UnitGateTests(unittest.TestCase):
    def test_prefix_dot_boundary_exact(self):
        self.assertTrue(pms.matches_prefix("mtp.layers.0.q_proj", "mtp."))
        self.assertTrue(pms.matches_prefix("mtp", "mtp."))          # bare key ok
        self.assertFalse(pms.matches_prefix("blocks.5.mtp.old.weight", "mtp."))
        self.assertFalse(pms.matches_prefix("model.mtp.extra", "mtp."))
        self.assertFalse(pms.matches_prefix("mtpx.norm", "mtp."))
        self.assertFalse(pms.matches_prefix("embed_tokens.weight", "mtp."))

    def test_revision_must_be_pinned_40hex(self):
        self.assertEqual(pms.validate_revision(REV), REV)
        for bad in ("main", "master", REV[:39], REV.upper(), REV + "0", ""):
            with self.assertRaises(pms.PrepareError):
                pms.validate_revision(bad)

    def test_base_url_tls_or_loopback(self):
        self.assertEqual(pms.validate_base_url("https://huggingface.co"),
                         "https://huggingface.co")
        self.assertEqual(pms.validate_base_url("http://127.0.0.1:8765"),
                         "http://127.0.0.1:8765")
        with self.assertRaises(pms.PrepareError):
            pms.validate_base_url("http://example.invalid")
        with self.assertRaises(pms.PrepareError):
            pms.validate_base_url("ftp://huggingface.co")

    def test_float_size_gate(self):
        pms.validate_header_tensors("t", {"mtp.a": {"dtype": "BF16", "shape": [4],
                                                    "data_offsets": [0, 8]}})
        with self.assertRaises(pms.PrepareError):        # numel x elsize != span
            pms.validate_header_tensors("t", {"mtp.a": {"dtype": "F32", "shape": [4],
                                                        "data_offsets": [0, 8]}})

    def test_degenerate_prefix_rejected(self):
        with self.assertRaises(pms.PrepareError):
            pms.matches_prefix("anything", ".")


class SelectorTests(unittest.TestCase):
    def test_no_substring_or_trunk_contamination(self):
        files, _p, weight_map = make_repo()
        sel = pms.Selector(json.loads(files[pms.DEFAULT_INDEX_FILE]), "mtp.")
        self.assertEqual(set(sel.selection), GOOD_TENSORS)
        for decoy in DECOYS:
            self.assertIn(decoy, weight_map)            # present in the official index...
            self.assertNotIn(decoy, sel.selection)      # ...but never selected

    def test_empty_selection_fails_closed(self):
        with self.assertRaises(pms.PrepareError):
            pms.Selector({"weight_map": {"embed_tokens.weight": "a.safetensors"}}, "mtp.")

    def test_bad_weight_map_entries(self):
        with self.assertRaises(pms.PrepareError):
            pms.Selector({"weight_map": {}}, "mtp.")
        with self.assertRaises(pms.PrepareError):
            pms.Selector({"weight_map": {"mtp.a": "model.bin"}}, "mtp.")


class ServerCase(unittest.TestCase):
    """Base: build the repo fixture, serve it over loopback, run the CLI."""

    mode = "normal"
    repo_kwargs = {}
    extra_argv = ()

    def setUp(self):
        self.files, self.payloads, self.weight_map = make_repo(**self.repo_kwargs)
        self.srv_ctx = FakeHF(self.files, mode=self.mode)
        self.base_url = self.srv_ctx.__enter__()
        self.addCleanup(self.srv_ctx.__exit__)
        self._tmp = tempfile.TemporaryDirectory(prefix="pmtp-test-")
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name) / "prepared"

    def clear_log(self):
        with self.srv_ctx._lock:
            self.srv_ctx.log.clear()

    def shard_requests(self):
        return [(p, r, s, n) for (p, r, s, n) in self.srv_ctx.log
                if p.split("?")[0].endswith(".safetensors")]

    def data_range_requests(self):
        """Range requests that are neither the 8-byte size probe nor the
        header body (start 8): i.e. actual tensor-payload fetches."""
        out = []
        for p, r, s, n in self.shard_requests():
            if r and not re.match(r"^bytes=0-7$", r) and not re.match(r"^bytes=8-\d+$", r):
                out.append((p, r, s, n))
        return out


class HappyPathTests(ServerCase):

    def test_end_to_end_byte_exact_roundtrip(self):
        rc, so, se = run_tool(self.base_url, self.out_dir,
                              extra=self.extra_argv + ("--chunk-bytes", "128"))
        self.assertEqual(rc, 0, msg=se)
        out_file = self.out_dir / pms.DEFAULT_OUTPUT_NAME
        self.assertTrue(out_file.exists())
        n, header, payload, raw = read_st(out_file)

        # names preserved exactly; ONLY official mtp.* keys made it in
        self.assertEqual(set(header), GOOD_TENSORS)
        self.assertNotIn("__metadata__", header)

        # every selected payload is BYTE-IDENTICAL to the official tensor bytes
        for name in GOOD_TENSORS:
            self.assertEqual(payload[name], self.payloads[name], name)
        # trunk/decoy payloads appear nowhere in the repackaged file
        for name in DECOYS:
            blob = self.payloads[name]
            self.assertNotIn(blob[: min(len(blob), 32)], raw, name)

        # valid padded header; data region contiguous from 0, no overlap
        self.assertEqual((8 + n) % 8, 0)
        spans = sorted(tuple(v["data_offsets"]) for v in header.values())
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(raw) - 8 - n)
        for (_b0, e0), (b1, _e1) in zip(spans, spans[1:]):
            self.assertLessEqual(e0, b1)

        # config.json and index saved BYTE-FOR-BYTE unmodified
        self.assertEqual((self.out_dir / pms.DEFAULT_CONFIG_FILE).read_bytes(),
                         self.files[pms.DEFAULT_CONFIG_FILE])
        self.assertEqual((self.out_dir / pms.DEFAULT_INDEX_FILE).read_bytes(),
                         self.files[pms.DEFAULT_INDEX_FILE])

        # no full-shard download ever happened: shard hits were 206 spans only
        shard_sizes = {name: len(self.files[name]) for name in SHARD_NAMES}
        for p, r, s, nbytes in self.shard_requests():
            self.assertEqual(s, 206, (p, r))
            fname = p.split("?")[0].rsplit("/", 1)[-1]
            self.assertLess(nbytes, shard_sizes[fname], "full-shard download")
        self.assertTrue(self.data_range_requests())   # payloads really were ranged

    def test_provenance_manifest_contents(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        manifest = json.loads((self.out_dir / pms.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["schema"], pms.SCHEMA)
        self.assertEqual(manifest["status"], "complete")
        ident = manifest["identity"]
        self.assertEqual(ident["repo"], REPO_ID)
        self.assertEqual(ident["revision"], REV)
        self.assertEqual(ident["tensor_prefix"], "mtp.")
        self.assertEqual(ident["base_url"], self.base_url)

        raw = (self.out_dir / pms.DEFAULT_OUTPUT_NAME).read_bytes()
        self.assertEqual(manifest["output"]["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(manifest["output"]["file_bytes"], len(raw))
        self.assertEqual(manifest["output"]["tensor_count"], len(GOOD_TENSORS))
        self.assertEqual(manifest["config"]["sha256"],
                         hashlib.sha256(self.files[pms.DEFAULT_CONFIG_FILE]).hexdigest())
        self.assertEqual(manifest["index"]["sha256"],
                         hashlib.sha256(self.files[pms.DEFAULT_INDEX_FILE]).hexdigest())

        # per-tensor: dtype/shape/source offsets + SHA256 of the fetched bytes
        for name, rec in manifest["tensors"].items():
            self.assertEqual(rec["shard"], self.weight_map[name])
            b, e = rec["data_offsets"]
            self.assertEqual(e - b, rec["bytes"])
            self.assertEqual(rec["sha256"],
                             hashlib.sha256(self.payloads[name]).hexdigest())
            expect_dtype = ("F32" if name == "mtp.layers.0.mlp.gate_proj.weight"
                            else "BF16")
            self.assertEqual(rec["dtype"], expect_dtype)
        self.assertEqual(manifest["expected_total_bytes"],
                         sum(rec["bytes"] for rec in manifest["tensors"].values()))

        # every referenced shard's FULL header JSON recorded with its hash
        for fname, rec in manifest["source_shards"].items():
            n = struct.unpack("<Q", self.files[fname][:8])[0]
            self.assertEqual(rec["header_json_bytes"], n)
            self.assertEqual(rec["header_sha256"],
                             hashlib.sha256(self.files[fname][8:8 + n]).hexdigest())
            self.assertEqual(rec["shard_total_bytes"], len(self.files[fname]))
            hdr = json.loads(rec["header"])
            # header lists every tensor of the shard, trunk keys included:
            for name, shard in self.weight_map.items():
                if shard == fname:
                    self.assertIn(name, hdr)
            if fname == "model-00003.safetensors":
                self.assertEqual(hdr["__metadata__"], {"format": "pt"})
            else:
                self.assertNotIn("__metadata__", hdr)

    def test_rerun_is_zero_network_idempotent(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        first = (self.out_dir / pms.DEFAULT_OUTPUT_NAME).read_bytes()
        self.clear_log()
        rc2, so2, se2 = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc2, 0, msg=se2)
        self.assertIn("already-complete", so2)
        # the prior manifest proves the pinned identity: NOTHING requested
        self.assertEqual(self.srv_ctx.log, [])
        self.assertEqual((self.out_dir / pms.DEFAULT_OUTPUT_NAME).read_bytes(), first)

    def test_in_progress_resume_refetches_nothing_rebuilds_deterministically(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        manifest_path = self.out_dir / pms.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        recorded_sha = manifest["output"]["sha256"]
        # simulate a crash right before the final manifest flip
        manifest["status"] = "in_progress"
        manifest["output"] = {}
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        self.clear_log()
        rc2, so2, se2 = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc2, 0, msg=se2)
        self.assertEqual(self.srv_ctx.log, [], "resume touched the network at all")
        manifest2 = json.loads(manifest_path.read_text())
        self.assertEqual(manifest2["status"], "complete")
        # repackaging from proven cache alone reproduces the identical bytes
        self.assertEqual(manifest2["output"]["sha256"], recorded_sha)

    def test_concurrent_jobs_deterministic_output(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        seq = (self.out_dir / pms.DEFAULT_OUTPUT_NAME).read_bytes()
        m1 = json.loads((self.out_dir / pms.MANIFEST_NAME).read_text())
        out2 = Path(self._tmp.name) / "prepared_jobs"
        rc2, so2, se2 = run_tool(self.base_url, out2, extra=("--jobs", "4"))
        self.assertEqual(rc2, 0, msg=se2)
        self.assertEqual((out2 / pms.DEFAULT_OUTPUT_NAME).read_bytes(), seq)
        m2 = json.loads((out2 / pms.MANIFEST_NAME).read_text())
        self.assertEqual(m1["output"]["sha256"], m2["output"]["sha256"])


class RefusalTests(ServerCase):

    def test_foreign_content_without_manifest_refused(self):
        self.out_dir.mkdir(parents=True)
        strays = self.out_dir / "someone_elses.safetensors"
        strays.write_bytes(b"do not touch me")
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertNotEqual(rc, 0)
        self.assertIn("provenance_manifest", se)
        self.assertEqual(strays.read_bytes(), b"do not touch me")

    def test_branch_revision_refused_before_any_request(self):
        rc, so, se = run_tool(self.base_url, self.out_dir, extra=("--revision", "main"))
        self.assertEqual(rc, 1)
        self.assertIn("40 lowercase hex", se)
        self.assertEqual(self.srv_ctx.log, [])

    def test_identity_change_across_runs_refused(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        before = (self.out_dir / pms.DEFAULT_OUTPUT_NAME).read_bytes()
        rc2, so2, se2 = run_tool(self.base_url, self.out_dir,
                                 extra=("--tensor-prefix", "mtp.layers."))
        self.assertNotEqual(rc2, 0)
        self.assertIn("different parameters", se2)
        self.assertEqual((self.out_dir / pms.DEFAULT_OUTPUT_NAME).read_bytes(), before)

    def test_tampered_output_not_silently_rewritten(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        out_file = self.out_dir / pms.DEFAULT_OUTPUT_NAME
        good = out_file.read_bytes()
        bad = bytearray(good)
        bad[-1] ^= 0xFF
        out_file.write_bytes(bytes(bad))
        rc2, so2, se2 = run_tool(self.base_url, self.out_dir)
        self.assertNotEqual(rc2, 0)
        self.assertIn("tampered", se2)
        self.assertEqual(out_file.read_bytes(), bytes(bad))   # still tampered on disk

    def test_tampered_cache_then_saved_config_refused(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        caches = sorted((self.out_dir / pms.CACHE_DIR_NAME / "tensors").glob("*.bin"))
        self.assertTrue(caches)
        c = caches[0]
        orig = c.read_bytes()
        bad = bytearray(orig)
        bad[0] ^= 0xFF
        c.write_bytes(bytes(bad))
        self.clear_log()
        rc2, so2, se2 = run_tool(self.base_url, self.out_dir)
        self.assertNotEqual(rc2, 0)
        self.assertIn("diverged", se2)
        self.assertEqual(self.srv_ctx.log, [],
                         "tool fetched over a divergent cache instead of refusing")
        c.write_bytes(orig)                       # restore cache
        (self.out_dir / pms.DEFAULT_CONFIG_FILE).write_bytes(b'{"tampered": 1}')
        rc3, so3, se3 = run_tool(self.base_url, self.out_dir)
        self.assertNotEqual(rc3, 0)
        self.assertIn("no longer matches", se3)


class RangeSemanticsBase(ServerCase):
    """Servers that ignore or lie about Range must be REJECTED, never used
    as an implicit full-shard fallback; no output may be produced."""

    def _check_rejected(self, expect_msg):
        rc, so, se = run_tool(self.base_url, self.out_dir,
                              extra=self.extra_argv + ("--chunk-bytes", "96"))
        self.assertNotEqual(rc, 0, msg=f"tool accepted mode={self.mode}")
        self.assertIn(expect_msg, se)
        self.assertFalse((self.out_dir / pms.DEFAULT_OUTPUT_NAME).exists())


class RangeIgnoredTests(RangeSemanticsBase):
    mode = "ignore_range"

    def test_lying_server_rejected(self):
        self._check_rejected("ignored the Range")


class BadContentRangeTests(RangeSemanticsBase):
    mode = "bad_content_range"

    def test_lying_server_rejected(self):
        self._check_rejected("does not match")


class TruncatedResponseTests(RangeSemanticsBase):
    mode = "truncate_cl"

    def test_lying_server_rejected(self):
        self._check_rejected("short read")


class EarlyCloseTests(RangeSemanticsBase):
    mode = "early_close"

    def test_lying_server_rejected(self):
        self._check_rejected("attempts exhausted")


class MissingShardTests(RangeSemanticsBase):
    mode = "not_found_shard2"

    def test_lying_server_rejected(self):
        self._check_rejected("404")


class FlakyRetryTests(ServerCase):
    """A transiently-corrupt first 206 must be RETRIED to exact bytes (retry
    path is real, not just exhaustion) and leave no temp files behind."""
    mode = "flaky_once"

    def test_transient_corruption_recovers_to_exact_payloads(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        _n, header, payload, _raw = read_st(self.out_dir / pms.DEFAULT_OUTPUT_NAME)
        self.assertEqual(set(header), GOOD_TENSORS)
        for name in GOOD_TENSORS:
            self.assertEqual(payload[name], self.payloads[name])
        strays = [p for p in self.out_dir.rglob("*")
                  if p.is_file() and (".part." in p.name or ".tmp." in p.name)]
        self.assertEqual(strays, [], "atomic-temp cleanup failed")
        m = json.loads((self.out_dir / pms.MANIFEST_NAME).read_text())
        self.assertEqual(m["status"], "complete")


class RedirectTests(ServerCase):
    """Redirects + cache-busting query strings must not break the pinned
    identity check, and the Range header must survive the hop."""
    mode = "redirect"

    def test_redirect_followed_and_payloads_still_exact(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        statuses = [s for (_p, _r, s, _n) in self.srv_ctx.log]
        self.assertIn(302, statuses)
        self.assertTrue(any("?cachebust=1" in p for (p, _r, _s, _n) in self.srv_ctx.log))
        # every range that got redirected still came back 206 with exact bytes
        self.assertTrue(all(s in (302, 206, 200)
                            for (_p, _r, s, _n) in self.srv_ctx.log))
        _n, _h, payload, _raw = read_st(self.out_dir / pms.DEFAULT_OUTPUT_NAME)
        for name in GOOD_TENSORS:
            self.assertEqual(payload[name], self.payloads[name])


class CorruptSelectionTests(ServerCase):
    """Non-float / zero-byte SELECTIONS must abort at plan time, before any
    tensor-payload Range request leaves the tool."""

    def _check_bad_selection(self, expect_msg):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertNotEqual(rc, 0)
        self.assertIn(expect_msg, se)
        self.assertEqual(self.data_range_requests(), [])
        self.assertFalse((self.out_dir / pms.DEFAULT_OUTPUT_NAME).exists())


class NonFloatSelectedTests(CorruptSelectionTests):
    repo_kwargs = {"extra": [(0, "mtp.bad.int", "I32", [4], 16)]}

    def test_bad_selection_aborts_before_payload_fetch(self):
        self._check_bad_selection("unquantized float")


class ZeroByteSelectedTests(CorruptSelectionTests):
    repo_kwargs = {"extra": [(2, "mtp.empty.gate", "BF16", [0], 0)]}

    def test_bad_selection_aborts_before_payload_fetch(self):
        self._check_bad_selection("zero-byte")


class TrunkIntIsFineTests(ServerCase):
    """Control for the test above: a non-float tensor OUTSIDE the mtp.*
    selection (trunk bookkeeping table) must not break a float-only run."""
    repo_kwargs = {"extra": [(1, "model.rotary.inv_freq_table", "I32", [4], 16)]}

    def test_trunk_nonfloat_does_not_affect_mtp_run(self):
        rc, so, se = run_tool(self.base_url, self.out_dir)
        self.assertEqual(rc, 0, msg=se)
        _n, header, payload, _raw = read_st(self.out_dir / pms.DEFAULT_OUTPUT_NAME)
        self.assertEqual(set(header), GOOD_TENSORS)
        for name in GOOD_TENSORS:
            self.assertEqual(payload[name], self.payloads[name])


class HeaderIntegrityTests(unittest.TestCase):
    """Direct-parse proofs for the malformed-header families (no network)."""

    @staticmethod
    def _parse(header_json: str, blob_len: int, limit: int = 1 << 20):
        data = det_bytes("blob", blob_len)
        body = header_json.encode("utf-8")
        body += b" " * ((-len(body)) % 8)
        raw = struct.pack("<Q", len(body)) + body + data
        n = struct.unpack("<Q", raw[:8])[0]
        return pms.ShardHeader.parse("x.safetensors", n, raw[8:8 + n], len(raw), limit)

    def test_overlap_rejected(self):
        h = ('{"mtp.a": {"dtype": "BF16", "shape": [8], "data_offsets": [0, 16]},'
             ' "mtp.b": {"dtype": "BF16", "shape": [8], "data_offsets": [8, 24]}}')
        with self.assertRaisesRegex(pms.PrepareError, "overlap"):
            self._parse(h, 40)

    def test_out_of_range_rejected(self):
        h = '{"mtp.a": {"dtype": "BF16", "shape": [100], "data_offsets": [0, 200]}}'
        with self.assertRaisesRegex(pms.PrepareError, "exceed data region"):
            self._parse(h, 40)

    def test_duplicate_keys_rejected(self):
        h = ('{"mtp.a": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},'
             ' "mtp.a": {"dtype": "BF16", "shape": [4], "data_offsets": [8, 16]}}')
        with self.assertRaisesRegex(pms.PrepareError, "duplicate"):
            self._parse(h, 24)

    def test_dtype_shape_span_mismatch_rejected(self):
        h = '{"mtp.a": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 9]}}'
        with self.assertRaisesRegex(pms.PrepareError, "size mismatch"):
            self._parse(h, 16)

    def test_negative_dim_rejected(self):
        h = '{"mtp.a": {"dtype": "BF16", "shape": [-4], "data_offsets": [0, 8]}}'
        with self.assertRaisesRegex(pms.PrepareError, "negative"):
            self._parse(h, 16)

    def test_bool_dim_rejected(self):
        h = '{"mtp.a": {"dtype": "BF16", "shape": [true], "data_offsets": [0, 8]}}'
        with self.assertRaisesRegex(pms.PrepareError, "not an int"):
            self._parse(h, 16)

    def test_header_limit_enforced(self):
        h = json.dumps({"mtp.a": {"dtype": "BF16", "shape": [16],
                                  "data_offsets": [0, 32]}})
        with self.assertRaisesRegex(pms.PrepareError, "exceeds"):
            self._parse(h, 100, limit=16)

    def test_bad_json_rejected(self):
        with self.assertRaisesRegex(pms.PrepareError, "invalid"):
            self._parse('{"mtp.a": ', 16)

    def test_header_beyond_eof_rejected(self):
        body = b'{"mtp.a": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}'
        raw = struct.pack("<Q", 128) + body + det_bytes("blob", 16)
        with self.assertRaisesRegex(pms.PrepareError, "runs past"):
            pms.ShardHeader.parse("x.safetensors", 128, raw[8:8 + len(body)],
                                  len(raw), 1 << 20)

    def test_truncated_body_vs_claimed_len(self):
        body = b'{"mtp.a": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}'
        raw = struct.pack("<Q", 128) + body + det_bytes("blob", 128 - len(body) + 16)
        supplied = raw[8:8 + 40]
        self.assertEqual(len(supplied), 40)
        with self.assertRaisesRegex(pms.PrepareError, "bytes !="):
            pms.ShardHeader.parse("x.safetensors", 128, supplied, len(raw), 1 << 20)


class HeaderLimitCliTests(ServerCase):
    def test_oversized_header_refused_without_body_download(self):
        rc, so, se = run_tool(self.base_url, self.out_dir,
                              extra=("--header-limit-bytes", "16"))
        self.assertNotEqual(rc, 0)
        self.assertIn("header-limit-bytes", se)
        # only 8-byte probes were made against shards; no header-body download
        for p, r, s, n in self.shard_requests():
            self.assertEqual(r, "bytes=0-7", (p, r))


class CLIInputTests(unittest.TestCase):
    def test_branch_revision_exit_code(self):
        so, se = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = pms.main(["--revision", "main", "--output-dir", "/tmp/pmtp-nope"])
        self.assertEqual(rc, 1)
        self.assertIn("40 lowercase hex", se.getvalue())

    def test_non_loopback_http_rejected_exit_code(self):
        so, se = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = pms.main(["--revision", REV, "--base-url", "http://10.0.0.9:80",
                           "--output-dir", "/tmp/pmtp-nope"])
        self.assertEqual(rc, 1)
        self.assertIn("loopback", se.getvalue())


class ListModeTests(ServerCase):
    def test_list_writes_nothing_and_selects_exactly(self):
        rc, so, se = run_tool(self.base_url, None, extra=("--list",))
        self.assertEqual(rc, 0, msg=se)
        self.assertIn("selected tensors: 4", so)
        for name in GOOD_TENSORS:
            self.assertIn(name, so)
        for decoy in DECOYS:
            self.assertNotIn(decoy, so)
        self.assertFalse(self.out_dir.exists())
        # only the index was fetched -- no shard, no config
        paths = {p.split("?")[0] for (p, _r, _s, _n) in self.srv_ctx.log}
        self.assertEqual(paths, {f"/Qwen/Qwen3.8-Flash-Next/resolve/{REV}/"
                                 f"{pms.DEFAULT_INDEX_FILE}"})


class OutputVerifierTests(unittest.TestCase):
    """verify_output_file must catch a corrupted repackage even if recorded
    hashes somehow agreed (defense in depth on reruns)."""

    def _write(self, td, header_dict, data: bytes) -> Path:
        p = Path(td) / "o.safetensors"
        body = json.dumps(header_dict).encode("utf-8")
        body += b" " * ((-len(body)) % 8)
        p.write_bytes(struct.pack("<Q", len(body)) + body + data)
        return p

    def test_good_passes(self):
        with tempfile.TemporaryDirectory(prefix="pmtp-verify-") as td:
            p = self._write(td, {"mtp.a": {"dtype": "BF16", "shape": [4],
                                           "data_offsets": [0, 8]}}, b"\x01" * 8)
            pms.verify_output_file(p, "o")

    def test_metadata_injected_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pmtp-verify-") as td:
            p = self._write(td, {"mtp.a": {"dtype": "BF16", "shape": [4],
                                           "data_offsets": [0, 8]},
                                 "__metadata__": {"format": "pt"}}, b"\x01" * 8)
            with self.assertRaisesRegex(pms.PrepareError, "__metadata__"):
                pms.verify_output_file(p, "o")

    def test_noncontiguous_data_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pmtp-verify-") as td:
            p = self._write(td, {"mtp.a": {"dtype": "BF16", "shape": [4],
                                           "data_offsets": [0, 8]}}, b"\x01" * 16)
            with self.assertRaisesRegex(pms.PrepareError, "contiguous"):
                pms.verify_output_file(p, "o")

    def test_nonfloat_dtype_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pmtp-verify-") as td:
            p = self._write(td, {"mtp.a": {"dtype": "I32", "shape": [2],
                                           "data_offsets": [0, 8]}}, b"\x01" * 8)
            with self.assertRaisesRegex(pms.PrepareError, "unquantized float"):
                pms.verify_output_file(p, "o")


if __name__ == "__main__":
    unittest.main()
