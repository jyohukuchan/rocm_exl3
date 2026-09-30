#!/usr/bin/env python3
"""Prepare an official, unquantized MTP-only safetensors source for the
controlled Qwen3.8 Flash Next native-MTP comparison (3-bit EXL3 vs
self-quantized 5-bit), for later consumption by util/convert_mtp.py.

What this does
--------------
Fetches ONLY the official ``mtp.*`` tensors from the pinned HF repo
(default ``Qwen/Qwen3.8-Flash-Next``) using HTTP Range requests against the
referenced safetensors shards (discovered through the official
``model.safetensors.index.json``), and repackages the fetched bytes UNCHANGED
into one local MTP-only safetensors file. Original tensor names are preserved
exactly as they appear in the official index. Shared embedding / LM-head
tensors are NOT acquired: the runtime shares the target model's unchanged
tables, and convert_mtp.py loads only the ``mtp`` component from this source
(any config-side prefix remap belongs downstream; names here stay byte- and
key-identical to the official checkpoint).

Also saved, byte-for-byte unmodified: the official ``config.json`` and the
official safetensors index, plus a provenance manifest
(``provenance_manifest.json``) recording repo/revision, each referenced
shard's full header JSON and its SHA256, per-tensor dtype/shape/source
offsets and SHA256 of the fetched bytes, total bytes, and SHA256 of the
output file, config and index. The manifest doubles as the resumable
state file, so a rerun after an interrupted fetch re-verifies hashes
instead of re-downloading.

Integrity rules (fail-closed, no silent overwrite)
--------------------------------------------------
* ``--revision`` is REQUIRED and must be exactly 40 lowercase hex chars. No
  branch names, no defaults: resume provenance depends on content-immutable
  pinning (current official revision at time of writing:
  de4b8e4d43b917e7706784d8bb445c9af86a3540 -- still must be passed).
* Range responses must be HTTP 206 with Content-Range exactly matching the
  requested ``begin-(end-1)`` span (and the known shard total when one has
  been established), and the delivered byte count must match exactly. A
  server that ignores the Range header (200 + whole file) is rejected;
  there is NO full-shard-download fallback.
* Redirects are followed, but the final URL must be https (loopback http
  only for the CPU test harness), on the base host or an
  ``*.huggingface.co``/``*.hf.co`` CDN host, or under the same pinned
  ``/repo/resolve/revision/`` path. ``Cache-Control: no-cache`` is sent so
  intermediaries cannot serve stale query-cached bodies.
* Shard headers are read with two bounded Range probes (the u64 length,
  then exactly the JSON body), capped by ``--header-limit-bytes``;
  duplicate JSON keys are rejected; offsets are validated for range and
  pairwise overlap against the real shard size; float dtypes must have
  numel*elsize == offset span. ``__metadata__`` is recorded for provenance
  but EXCLUDED from the output header.
* Every selected tensor must be an unquantized float dtype (BF16/F16/F32/
  F64) with a nonzero payload; anything else aborts BEFORE any tensor
  bytes are requested. Selection uses a dot-boundary-exact prefix
  (default ``mtp.``): ``x.mtp.y`` (substring inside a trunk key),
  ``model.mtp.y`` and ``mtpx.y`` are never pulled in.
* ``--output-dir`` must be fresh, empty, or a proven-compatible resume
  directory (its manifest matches every requested parameter, and cached /
  saved artifacts hash to the recorded values). A tampered output or cache,
  a manifest from different parameters, or foreign pre-existing files
  abort. A completed verified manifest makes reruns zero-network.
* Downloads stream in bounded chunks (``--chunk-bytes``) into atomic temp
  files (rename at the end); transient failures (transport errors, HTTP
  5xx/429/408, malformed range responses) retry with capped exponential
  backoff up to ``--retries``. Per-tensor cache in ``fetch_cache/tensors``
  makes every rerun resumable at exact boundaries. Fetching is sequential
  by default (``--jobs 1``, the recommended reliability-first path);
  ``--jobs`` up to 8 runs modest concurrent fetches writing the same
  manifest under a lock.

Usage (host, CPU-only, network read of pinned artifacts only):

    python3 rocm_tools/rdna2/prepare_mtp_source.py \
        --repo Qwen/Qwen3.8-Flash-Next \
        --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
        --output-dir /path/to/rocm_exl3/models/mtp_source_qwen38flashnext

Inspection only (fetches the index, prints the selection, writes nothing to
the output dir): add ``--list``.

Exit codes: 0 on success (including verified-complete reruns), 1 on any
fail-closed error, 130 on interrupt (rerun resumes). Stdlib only; torch and
the safetensors package are deliberately NOT imported so this runs wherever
the repo's CPU tests run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import struct
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

SCHEMA = "rdna2-mtp-source-provenance/1"
DEFAULT_REPO = "Qwen/Qwen3.8-Flash-Next"
DEFAULT_BASE_URL = "https://huggingface.co"
DEFAULT_PREFIX = "mtp."
DEFAULT_INDEX_FILE = "model.safetensors.index.json"
DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_OUTPUT_NAME = "mtp_source.safetensors"
DEFAULT_REVISION_HINT = "de4b8e4d43b917e7706784d8bb445c9af86a3540"

MANIFEST_NAME = "provenance_manifest.json"
CACHE_DIR_NAME = "fetch_cache"

USER_AGENT = "rocm-exl3-prepare-mtp-source/1.0"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)

# Unquantized wide-float dtypes accepted for the MTP source. Ints, bools,
# fp8/fp4 and unknown dtype strings are quantized/compressed: selection of
# any such tensor aborts before a single payload byte is requested.
FLOAT_DTYPES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}

# Redirect targets allowed besides the base host itself and any final URL
# that still sits under the pinned /repo/resolve/revision/ path.
ALLOWED_REDIRECT_DOMAINS = ("huggingface.co", "hf.co")


class PrepareError(RuntimeError):
    """Fatal, fail-closed error with a user-facing message."""


class TransientNetworkError(PrepareError):
    """Retryable transport/response problem (including a server that
    mis-handles a Range request)."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def matches_prefix(name: str, prefix: str) -> bool:
    """Dot-boundary-exact prefix match: with the default ``mtp.`` a key is
    selected iff it is ``mtp`` or starts with ``mtp.`` -- never ``x.mtp.y``
    (substring inside a trunk key) and never ``mtpx.y``."""
    head = prefix.rstrip(".")
    if not head:
        raise PrepareError(f"tensor prefix {prefix!r} degenerates to an empty head")
    return name == head or name.startswith(head + ".")


def validate_revision(revision: str) -> str:
    if not REVISION_RE.match(revision):
        raise PrepareError(
            f"--revision must be exactly 40 lowercase hex characters (a full commit "
            f"SHA); got {revision!r}. Branch names and short SHAs are rejected: "
            f"resume provenance depends on content-immutable pinning. Current "
            f"official revision hint: {DEFAULT_REVISION_HINT}")
    return revision


def _is_loopback_host(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "[::1]")


def validate_base_url(base_url: str) -> str:
    p = urllib.parse.urlsplit(base_url)
    if p.scheme not in ("http", "https") or not p.netloc:
        raise PrepareError(f"--base-url must be an absolute http(s) URL, got {base_url!r}")
    if p.scheme == "http" and not _is_loopback_host((p.hostname or "").lower()):
        raise PrepareError(
            f"refusing plain http:// for non-loopback host {p.hostname!r}: pinned-"
            f"identity provenance requires TLS outside the CPU test harness")
    return base_url.rstrip("/")


def quote_repo_path(path: str) -> str:
    return "/".join(urllib.parse.quote(seg, safe="") for seg in path.split("/"))


class HttpSource:
    """Range-capable reader over HF ``resolve`` URLs. urllib preserves the
    Range header across redirects, but every response is validated anyway
    (206 + exact Content-Range + exact byte count), so a redirect that
    lands somewhere Range-ignoring is still rejected."""

    def __init__(self, base_url: str, repo: str, revision: str, *, timeout: float,
                 retries: int, backoff_max: float, chunk_bytes: int,
                 token: str | None = None, quiet: bool = False):
        self.base_url = validate_base_url(base_url)
        self.repo = repo
        self.revision = validate_revision(revision)
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.backoff_max = float(backoff_max)
        self.chunk_bytes = max(64, int(chunk_bytes))
        self.token = token
        self.quiet = quiet
        self.final_hosts: dict[str, str] = {}
        self.base_host = (urllib.parse.urlsplit(self.base_url).hostname or "").lower()
        self.allow_http = _is_loopback_host(self.base_host)
        self._print_lock = threading.Lock()

    def log(self, msg: str) -> None:
        if not self.quiet:
            with self._print_lock:
                print(msg, file=sys.stderr, flush=True)

    def url_for(self, path: str) -> str:
        return (f"{self.base_url}/{quote_repo_path(self.repo)}/resolve/"
                f"{self.revision}/{quote_repo_path(path)}")

    def _check_final_url(self, final_url: str, path: str, what: str) -> None:
        f = urllib.parse.urlsplit(final_url)
        if f.scheme not in ("http", "https"):
            raise PrepareError(f"{what}: unsupported redirect scheme {f.scheme!r}")
        if f.scheme == "http" and not self.allow_http:
            raise PrepareError(f"{what}: redirect left TLS: {final_url!r}")
        host = (f.hostname or "").lower()
        pinned_prefix = f"/{quote_repo_path(self.repo)}/resolve/{self.revision}/"
        ok_host = (host == self.base_host
                   or any(host == d or host.endswith("." + d)
                          for d in ALLOWED_REDIRECT_DOMAINS))
        if not (ok_host or f.path.startswith(pinned_prefix)):
            raise PrepareError(
                f"{what}: redirect to unexpected host/path ({final_url!r}); refusing "
                f"to fetch outside the pinned repo/revision identity")
        self.final_hosts.setdefault(path, host)

    def _open(self, url: str, begin: int | None, end: int | None, what: str):
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream",
            "Cache-Control": "no-cache",
        }
        if begin is not None:
            headers["Range"] = f"bytes={begin}-{end - 1}"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PrepareError(f"{what}: HTTP {e.code} -- private repo or bad token?")
            if e.code == 404:
                raise PrepareError(f"{what}: HTTP 404 at {url!r} "
                                   f"(check --repo/--revision/--index-file)")
            if e.code == 416:
                raise PrepareError(f"{what}: HTTP 416 Range Not Satisfiable -- the file "
                                   f"is smaller than the offsets the header claimed")
            if e.code in (408, 429) or 500 <= e.code < 600:
                raise TransientNetworkError(f"{what}: HTTP {e.code}")
            raise PrepareError(f"{what}: HTTP {e.code} {e.reason}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise TransientNetworkError(f"{what}: transport error: {e}")
        return resp

    def _stream_to(self, resp, tmp: Path, length: int | None) -> tuple[int, str]:
        """Read resp into tmp (at most chunk_bytes resident); over-delivery
        and short reads are both failures. Returns (bytes, sha256)."""
        h = hashlib.sha256()
        got = 0
        cl = resp.headers.get("Content-Length")
        try:
            with open(tmp, "wb") as out:
                while True:
                    buf = resp.read(self.chunk_bytes)
                    if not buf:
                        break
                    got += len(buf)
                    h.update(buf)
                    out.write(buf)
                out.flush()
                os.fsync(out.fileno())
        except (OSError, http.client.HTTPException) as e:
            # HTTPException covers IncompleteRead/RemoteDisconnected when the
            # server advertises a span and then dies mid-stream.
            raise TransientNetworkError(f"aborted mid-stream: {e}")
        if length is not None:
            if got > length:
                raise TransientNetworkError(
                    f"server sent MORE than the requested {length} bytes ({got})")
            if got != length:
                raise TransientNetworkError(
                    f"short read: got {got} bytes, expected exactly {length}")
        if cl is not None:
            try:
                cl_i = int(cl)
            except ValueError:
                cl_i = None
            if cl_i is not None and got != cl_i:
                raise TransientNetworkError(f"Content-Length {cl_i} but received {got}")
        return got, h.hexdigest()

    def fetch(self, dest: Path, path: str, begin: int | None = None,
              end: int | None = None, *, expect_total: int | None = None,
              what: str) -> dict:
        """Fetch [begin, end) (whole file if begin is None) into ``dest``
        atomically. Returns {"bytes", "sha256", "content_total"}, where
        content_total is the server-reported full size from Content-Range."""
        url = self.url_for(path)
        length = None if begin is None else end - begin
        if begin is not None and length <= 0:
            raise PrepareError(f"{what}: non-positive range length {length}")
        tmp = dest.parent / f"{dest.name}.part.{uuid.uuid4().hex}"
        last_err: Exception | None = None
        try:
            for attempt in range(1, self.retries + 1):
                try:
                    result = self._fetch_once(url, path, begin, end, length,
                                              expect_total, tmp, what)
                    os.replace(tmp, dest)
                    return result
                except TransientNetworkError as e:
                    # (checked BEFORE PrepareError: transient IS-a PrepareError,
                    #  and only transient failures are retryable)
                    last_err = e
                    if attempt >= self.retries:
                        break
                    delay = min(2.0 ** attempt, self.backoff_max)
                    self.log(f"   [retry {attempt}/{self.retries - 1}] {what}: {e} "
                             f"(sleep {delay:.1f}s)")
                    time.sleep(delay)
                except PrepareError:
                    raise
            raise PrepareError(f"{what}: {self.retries} attempts exhausted: {last_err}")
        finally:
            try:
                if tmp.exists():
                    os.remove(tmp)
            except OSError:
                pass

    def _fetch_once(self, url, path, begin, end, length, expect_total, tmp,
                    what) -> dict:
        resp = self._open(url, begin, end, what)
        with resp:
            self._check_final_url(resp.geturl(), path, what)
            status = getattr(resp, "status", None) or resp.getcode()
            content_total: int | None = None
            if begin is None:
                if status != 200:
                    raise TransientNetworkError(
                        f"{what}: whole-file fetch returned HTTP {status}, expected 200")
            else:
                if status != 206:
                    raise TransientNetworkError(
                        f"{what}: server ignored the Range header (HTTP {status}, "
                        f"expected 206); no full-download fallback is allowed")
                m = CONTENT_RANGE_RE.match((resp.headers.get("Content-Range") or "").strip())
                if not m:
                    raise TransientNetworkError(
                        f"{what}: missing/unparseable Content-Range on 206: "
                        f"{resp.headers.get('Content-Range')!r}")
                cs, ce, ct = int(m.group(1)), int(m.group(2)), int(m.group(3))
                content_total = ct
                if cs != begin or ce != end - 1:
                    raise TransientNetworkError(
                        f"{what}: Content-Range bytes {cs}-{ce}/{ct} does not match "
                        f"requested {begin}-{end - 1}")
                if ct < end:
                    raise TransientNetworkError(
                        f"{what}: server reports total {ct} < requested end {end}")
                if expect_total is not None and ct != expect_total:
                    raise TransientNetworkError(
                        f"{what}: Content-Range total {ct} != known shard size "
                        f"{expect_total}")
            got, sha_hex = self._stream_to(resp, tmp, length)
        return {"bytes": got, "sha256": sha_hex, "content_total": content_total}


def unpack_u64(b: bytes, what: str) -> int:
    if len(b) < 8:
        raise PrepareError(f"{what}: truncated u64 length prefix")
    return struct.unpack("<Q", b[:8])[0]


def pack_u64(n: int) -> bytes:
    return struct.pack("<Q", n)


def validate_shape(shape, name: str, where: str) -> int:
    if not isinstance(shape, list):
        raise PrepareError(f"{where}: tensor {name!r}: shape must be a list")
    numel = 1
    for d in shape:
        if isinstance(d, bool) or not isinstance(d, int):
            raise PrepareError(f"{where}: tensor {name!r}: shape dim {d!r} is not an int")
        if d < 0:
            raise PrepareError(f"{where}: tensor {name!r}: negative shape dim {d}")
        numel *= d
    return numel


def validate_header_tensors(where: str, obj: dict) -> dict:
    """Structural validation of a parsed safetensors tensor map (with
    __metadata__ already popped). Float dtypes must satisfy
    numel * elsize == offset span; non-float entries are kept (the selection
    gate rejects them) but still need sane offset types."""
    tensors: dict[str, dict] = {}
    for name, info in obj.items():
        if not isinstance(name, str) or not name:
            raise PrepareError(f"{where}: invalid tensor key {name!r}")
        if not isinstance(info, dict):
            raise PrepareError(f"{where}: tensor {name!r}: entry is not an object")
        dtype = info.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise PrepareError(f"{where}: tensor {name!r}: missing dtype")
        numel = validate_shape(info.get("shape"), name, where)
        offs = info.get("data_offsets")
        if (not isinstance(offs, list) or len(offs) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in offs)):
            raise PrepareError(f"{where}: tensor {name!r}: bad data_offsets {offs!r}")
        begin, end = offs
        if begin < 0 or end < begin:
            raise PrepareError(f"{where}: tensor {name!r}: bad offsets [{begin}, {end}]")
        elsize = FLOAT_DTYPES.get(dtype)
        if elsize is not None and numel * elsize != end - begin:
            raise PrepareError(
                f"{where}: tensor {name!r}: dtype/shape size mismatch: numel {numel} "
                f"x {elsize} != span {end - begin}")
        tensors[name] = {"dtype": dtype, "shape": list(info["shape"]),
                         "data_offsets": [begin, end]}
    return tensors


def check_no_overlap(where: str, intervals) -> None:
    """intervals: iterable of (begin, end, name). Zero-length spans never
    collide (the comparison ``begin < prev_end`` is strict)."""
    prev_end, prev_name = None, None
    for begin, end, name in sorted(intervals):
        if prev_end is not None and begin < prev_end:
            raise PrepareError(f"{where}: tensors {prev_name!r} and {name!r} overlap "
                               f"at {begin} < {prev_end}")
        prev_end, prev_name = end, name


class ShardHeader:
    """A parsed, fully validated safetensors shard header plus provenance.
    ``body`` is the header JSON bytes WITHOUT the 8-byte length prefix."""

    def __init__(self, filename, header_len, body, total_size, tensors, metadata):
        self.filename = filename
        self.header_len = header_len
        self.body = body
        self.total_size = total_size
        self.data_start = 8 + header_len
        self.tensors = tensors
        self.metadata = metadata
        self.header_sha256 = sha256_bytes(body)

    @classmethod
    def parse(cls, filename: str, header_len: int, body: bytes, total_size: int,
              header_limit: int) -> "ShardHeader":
        if header_len == 0:
            raise PrepareError(f"shard {filename}: zero-length safetensors header")
        if header_len > header_limit:
            raise PrepareError(f"shard {filename}: header length {header_len} exceeds "
                               f"--header-limit-bytes {header_limit} (no unbounded parse)")
        if 8 + header_len > total_size:
            raise PrepareError(f"shard {filename}: header length {header_len} runs past "
                               f"EOF (shard size {total_size})")
        if len(body) != header_len:
            raise PrepareError(f"shard {filename}: header body {len(body)} bytes != "
                               f"claimed {header_len}")

        def _no_dup(pairs):
            seen = set()
            for k, _ in pairs:
                if k in seen:
                    raise PrepareError(f"shard {filename}: duplicate JSON key {k!r}")
                seen.add(k)
            return dict(pairs)

        try:
            obj = json.loads(body.decode("utf-8"), object_pairs_hook=_no_dup)
        except UnicodeDecodeError as e:
            raise PrepareError(f"shard {filename}: header JSON not valid UTF-8: {e}")
        except json.JSONDecodeError as e:
            raise PrepareError(f"shard {filename}: header JSON invalid: {e}")
        if not isinstance(obj, dict):
            raise PrepareError(f"shard {filename}: header JSON is not an object")
        metadata = obj.pop("__metadata__", None)
        if metadata is not None and not isinstance(metadata, dict):
            raise PrepareError(f"shard {filename}: __metadata__ is not an object")
        tensors = validate_header_tensors(f"shard {filename}", obj)
        check_no_overlap(f"shard {filename}",
                         [(i["data_offsets"][0], i["data_offsets"][1], n)
                          for n, i in tensors.items()])
        data_len = total_size - (8 + header_len)
        for n, i in tensors.items():
            if i["data_offsets"][1] > data_len:
                raise PrepareError(
                    f"shard {filename}: tensor {n!r} offsets {i['data_offsets']} "
                    f"exceed data region 0..{data_len}")
        return cls(filename, header_len, body, total_size, tensors, metadata)


    def abs_range(self, begin: int, end: int) -> tuple[int, int]:
        """Header data_offsets are relative to the data region (which starts
        right after the 8-byte length prefix + header JSON); convert to
        absolute shard-file bytes for Range requests."""
        return self.data_start + begin, self.data_start + end


class Selector:
    """Turns the official index weight_map into a validated, dot-exact
    selection of prefix keys and the shards that hold them."""

    def __init__(self, index_obj: dict, prefix: str):
        if not isinstance(index_obj, dict):
            raise PrepareError("index: not a JSON object")
        wm = index_obj.get("weight_map")
        if not isinstance(wm, dict) or not wm:
            raise PrepareError("index: missing or empty weight_map")
        self.prefix = prefix
        self.selection: dict[str, str] = {}
        for name, shard in sorted(wm.items()):
            if not isinstance(name, str) or not isinstance(shard, str):
                raise PrepareError(f"index: bad weight_map entry {name!r}: {shard!r}")
            if matches_prefix(name, prefix):
                if not shard.endswith(".safetensors"):
                    raise PrepareError(f"index: {name!r} maps to non-safetensors "
                                       f"file {shard!r}")
                self.selection[name] = shard
        if not self.selection:
            raise PrepareError(
                f"index: NO keys match tensor prefix {prefix!r} (dot-boundary-exact). "
                f"Check the official naming with --list; substring matching is "
                f"refused because it would pull trunk keys.")
        self.shards: dict[str, list[str]] = {}
        for name, shard in self.selection.items():
            self.shards.setdefault(shard, []).append(name)


def copy_bytes(src: Path, dst: Path) -> None:
    tmp = dst.parent / f"{dst.name}.tmp.{uuid.uuid4().hex}"
    with open(src, "rb") as f, open(tmp, "wb") as g:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            g.write(chunk)
        g.flush()
        os.fsync(g.fileno())
    os.replace(tmp, dst)


def verify_output_file(path: Path, what: str) -> None:
    """Re-parse an output this tool wrote and enforce its invariants:
    valid padded header, no __metadata__, floats only, contiguous, in range."""
    size = path.stat().st_size
    if size < 8:
        raise PrepareError(f"{what}: file too small for a u64 header length")
    with open(path, "rb") as f:
        header_len = unpack_u64(f.read(8), what)
        if header_len == 0 or 8 + header_len > size:
            raise PrepareError(f"{what}: bad header length {header_len} for size {size}")
        if (8 + header_len) % 8 != 0:
            raise PrepareError(f"{what}: header block is not 8-byte padded")
        body = f.read(header_len)
        try:
            obj = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise PrepareError(f"{what}: header JSON invalid: {e}")
    if not isinstance(obj, dict):
        raise PrepareError(f"{what}: header JSON is not an object")
    if "__metadata__" in obj:
        raise PrepareError(f"{what}: __metadata__ must be excluded from the output")
    tensors = validate_header_tensors(what, obj)
    if not tensors:
        raise PrepareError(f"{what}: header lists no tensors")
    for n, i in tensors.items():
        if i["dtype"] not in FLOAT_DTYPES:
            raise PrepareError(f"{what}: tensor {n!r}: dtype {i['dtype']} in output "
                               f"is not an unquantized float")
    check_no_overlap(what, [(i["data_offsets"][0], i["data_offsets"][1], n)
                            for n, i in tensors.items()])
    data_len = size - 8 - header_len
    used_end = max(i["data_offsets"][1] for i in tensors.values())
    starts = sorted(i["data_offsets"][0] for i in tensors.values())
    if starts[0] != 0 or used_end != data_len:
        raise PrepareError(f"{what}: data region not contiguous "
                           f"(min start {starts[0]}, last end {used_end}, "
                           f"region {data_len})")


class MtpPreparer:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.output_dir)
        self.cache_dir = self.out_dir / CACHE_DIR_NAME
        self.meta_dir = self.cache_dir / "_meta"
        self.headers_dir = self.cache_dir / "_headers"
        self.tensors_dir = self.cache_dir / "tensors"
        self.manifest_path = self.out_dir / MANIFEST_NAME
        self.output_path = self.out_dir / args.output_name
        self.identity = {
            "base_url": args.base_url,
            "repo": args.repo,
            "revision": args.revision,
            "tensor_prefix": args.tensor_prefix,
            "index_file": args.index_file,
            "config_file": args.config_file,
            "output_name": args.output_name,
        }
        self.http = HttpSource(
            args.base_url, args.repo, args.revision,
            timeout=args.timeout, retries=args.retries,
            backoff_max=args.backoff_max, chunk_bytes=args.chunk_bytes,
            token=args.hf_token, quiet=args.quiet)
        self._lock = threading.Lock()

    def _cache_key(self, shard: str, name: str) -> str:
        material = "\n".join([self.identity["repo"], self.identity["revision"],
                              shard, name]).encode("utf-8")
        return sha256_bytes(material)

    def _write_json_atomic(self, path: Path, obj: dict) -> None:
        tmp = path.parent / f"{path.name}.tmp.{uuid.uuid4().hex}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _save_manifest(self, manifest: dict) -> None:
        with self._lock:
            self._write_json_atomic(self.manifest_path, manifest)

    def _load_existing_manifest(self):
        if not self.manifest_path.exists():
            return None
        try:
            m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise PrepareError(f"existing manifest {self.manifest_path} unreadable: {e}")
        if not isinstance(m, dict) or m.get("schema") != SCHEMA:
            raise PrepareError(f"existing manifest {self.manifest_path} is not a "
                               f"{SCHEMA} document (foreign artifact -- refusing)")
        got = {k: m.get("identity", {}).get(k) for k in self.identity}
        if got != self.identity:
            diff = {k: (got[k], v) for k, v in self.identity.items() if got[k] != v}
            raise PrepareError(
                f"existing manifest was prepared with different parameters: {diff}. "
                f"Refusing to resume across identities -- use a fresh --output-dir.")
        return m

    def _init_manifest(self) -> dict:
        return {
            "schema": SCHEMA,
            "identity": dict(self.identity),
            "status": "in_progress",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {},
            "index": {},
            "source_shards": {},
            "tensors": {},
            "output": {},
        }

    # ---------- small artifacts (index / config) ----------
    def ensure_artifact(self, rel_path: str, key: str, manifest: dict) -> dict:
        """Keep the EXACT official bytes in out_dir (saved unmodified) and in
        the _meta cache, with a manifest-proven SHA256. No re-download when
        cache + manifest already agree. Returns the parsed JSON object."""
        name = Path(rel_path).name
        saved = self.out_dir / name
        cached = self.meta_dir / (name + ".official")
        rec = manifest.setdefault(key, {})
        expect_sha = rec.get("sha256")
        if expect_sha and cached.exists() and sha256_file(cached) == expect_sha:
            res = {"bytes": cached.stat().st_size, "sha256": expect_sha}
        else:
            res = self.http.fetch(cached, rel_path, what=f"official {key} ({rel_path})")
            if expect_sha and res["sha256"] != expect_sha:
                raise PrepareError(
                    f"official {rel_path} hashes to {res['sha256']} but the manifest "
                    f"records {expect_sha}: content changed at a PINNED revision -- "
                    f"refusing (this must not happen; verify repo/revision)")
        if saved.exists():
            cur = sha256_file(saved)
            if cur != res["sha256"]:
                raise PrepareError(
                    f"{saved} exists with hash {cur}, but the official bytes hash to "
                    f"{res['sha256']}: refusing silent overwrite of an inconsistent "
                    f"existing output")
        else:
            copy_bytes(cached, saved)
        rec["path"] = rel_path
        rec["sha256"] = res["sha256"]
        rec["bytes"] = res["bytes"]
        try:
            obj = json.loads(cached.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise PrepareError(f"official {rel_path} is not valid JSON: {e}")
        if not isinstance(obj, dict):
            raise PrepareError(f"official {rel_path} is not a JSON object")
        return obj

    # ---------- shard headers ----------
    def ensure_shard_header(self, shard: str, manifest: dict) -> ShardHeader:
        rec = manifest["source_shards"].get(shard)
        body_path = self.headers_dir / (self._cache_key(shard, "__header__") + ".json")
        if rec and body_path.exists():
            body = body_path.read_bytes()
            if sha256_bytes(body) == rec["header_sha256"]:
                return ShardHeader.parse(shard, rec["header_json_bytes"], body,
                                         rec["shard_total_bytes"],
                                         self.args.header_limit_bytes)
            self.http.log(f"   [header cache invalid] {shard}: refetching")
        probe_path = self.headers_dir / (self._cache_key(shard, "__probe__") + ".len")
        probe = self.http.fetch(probe_path, shard, 0, 8,
                                what=f"shard {shard} size probe")
        total = probe["content_total"]
        if total is None:
            raise PrepareError(f"shard {shard}: probe response carried no total size")
        if rec and total != rec["shard_total_bytes"]:
            raise PrepareError(f"shard {shard}: size {total} differs from manifest "
                               f"record {rec['shard_total_bytes']} at a pinned revision")
        header_len = unpack_u64(probe_path.read_bytes(), f"shard {shard} probe")
        if header_len > self.args.header_limit_bytes:
            # gate BEFORE the body download: never pull an unbounded header
            raise PrepareError(f"shard {shard}: header length {header_len} exceeds "
                               f"--header-limit-bytes {self.args.header_limit_bytes}")
        if 8 + header_len > total:
            raise PrepareError(f"shard {shard}: header {header_len} bytes beyond file "
                               f"size {total}")
        got = self.http.fetch(body_path, shard, 8, 8 + header_len,
                              expect_total=total, what=f"shard {shard} header")
        if rec and got["sha256"] != rec["header_sha256"]:
            raise PrepareError(
                f"shard {shard}: header hash {got['sha256']} changed from manifest "
                f"record {rec['header_sha256']} at a pinned revision -- refusing")
        hdr = ShardHeader.parse(shard, header_len, body_path.read_bytes(), total,
                                self.args.header_limit_bytes)
        manifest["source_shards"][shard] = {
            "header_sha256": hdr.header_sha256,
            "header_json_bytes": hdr.header_len,
            "shard_total_bytes": hdr.total_size,
            "data_start": hdr.data_start,
            "resolve_path": shard,
            "resolved_host": self.http.final_hosts.get(shard),
            "metadata": hdr.metadata,
            "header": hdr.body.decode("utf-8"),
        }
        self._save_manifest(manifest)
        return hdr

    # ---------- plan ----------
    def build_plan(self, manifest: dict) -> tuple[Selector, dict[str, ShardHeader]]:
        idx_obj = self.ensure_artifact(self.args.index_file, "index", manifest)
        cfg_obj = self.ensure_artifact(self.args.config_file, "config", manifest)
        if "architectures" not in cfg_obj and "model_type" not in cfg_obj:
            raise PrepareError("official config.json lacks architectures/model_type -- "
                               "not a model config?")
        sel = Selector(idx_obj, self.args.tensor_prefix)
        meta = idx_obj.get("metadata")
        manifest["index"]["metadata"] = meta if isinstance(meta, dict) else {}
        manifest["index"]["selected_tensors"] = len(sel.selection)
        headers = {shard: self.ensure_shard_header(shard, manifest)
                   for shard in sorted(sel.shards)}
        n_bytes = 0
        for name, shard in sorted(sel.selection.items()):
            ti = headers[shard].tensors.get(name)
            if ti is None:
                raise PrepareError(f"index/header disagree: {name!r} not in shard "
                                   f"{shard} header")
            if ti["dtype"] not in FLOAT_DTYPES:
                raise PrepareError(
                    f"tensor {name!r}: dtype {ti['dtype']} is not an unquantized "
                    f"float (BF16/F16/F32/F64): the official MTP source must be "
                    f"unquantized -- aborting before any payload download")
            b, e = ti["data_offsets"]
            if e - b <= 0:
                raise PrepareError(f"tensor {name!r}: zero-byte payload, "
                                   f"unsupported here")
            rec = manifest["tensors"].setdefault(name, {})
            rec.update({"name": name, "shard": shard, "dtype": ti["dtype"],
                        "shape": ti["shape"], "data_offsets": [b, e],
                        "bytes": e - b})
            n_bytes += e - b
        manifest["expected_total_bytes"] = n_bytes
        self._save_manifest(manifest)
        self.http.log(f" -- selected {len(sel.selection)} tensors "
                      f"({n_bytes} bytes) across {len(sel.shards)} shard(s) of "
                      f"{self.args.repo}@{self.args.revision[:12]}")
        return sel, headers

    # ---------- per-tensor fetch ----------
    def fetch_tensor(self, manifest: dict, name: str, shard: str,
                     hdr: ShardHeader) -> None:
        rec = manifest["tensors"][name]
        begin, end = rec["data_offsets"]
        expected = end - begin
        if rec["dtype"] not in FLOAT_DTYPES:
            raise PrepareError(f"tensor {name!r}: non-float dtype {rec['dtype']} "
                               f"reached the fetch stage")
        prior_sha = rec.get("sha256")
        cache = self.tensors_dir / (self._cache_key(shard, name) + ".bin")
        if cache.exists():
            csha = sha256_file(cache)
            if prior_sha and csha == prior_sha and cache.stat().st_size == expected:
                self.http.log(f"   [cache hit] {name} ({expected} bytes)")
                return
            # Corrupt/orphan cache: drop and refetch. At a pinned revision a
            # refetch MUST reproduce prior_sha (enforced below), so a disk
            # blemish can never silently rewrite provenance.
            self.http.log(f"   [cache invalid] {name}: refetching")
            cache.unlink(missing_ok=True)
        a_begin, a_end = hdr.abs_range(begin, end)
        got = self.http.fetch(cache, shard, a_begin, a_end, expect_total=hdr.total_size,
                              what=f"tensor {name} from {shard}")
        if got["bytes"] != expected:
            raise PrepareError(f"tensor {name!r}: fetched {got['bytes']} bytes, "
                               f"expected {expected}")
        if prior_sha and got["sha256"] != prior_sha:
            raise PrepareError(
                f"tensor {name!r}: bytes at a pinned revision hash to "
                f"{got['sha256']} but the manifest recorded {prior_sha} -- refusing")
        rec["sha256"] = got["sha256"]
        self._save_manifest(manifest)
        self.http.log(f"   [fetched] {name}: {expected} bytes {got['sha256'][:12]}...")

    # ---------- output ----------
    def build_output(self, manifest: dict, sel: Selector) -> None:
        names = sorted(sel.selection)
        plan, cursor = [], 0
        for n in names:
            rec = manifest["tensors"][n]
            cache = self.tensors_dir / (self._cache_key(rec["shard"], n) + ".bin")
            if not cache.exists() or cache.stat().st_size != rec["bytes"] \
                    or sha256_file(cache) != rec.get("sha256"):
                raise PrepareError(f"internal: cache for {n!r} not present/verified "
                                   f"at packaging time")
            plan.append((n, rec, cursor, cursor + rec["bytes"], cache))
            cursor += rec["bytes"]
        check_no_overlap("output plan", [(b, e, n) for n, _, b, e, _ in plan])
        header = {n: {"dtype": rec["dtype"], "shape": rec["shape"],
                      "data_offsets": [b, e]}
                  for n, rec, b, e, _ in plan}
        body = json.dumps(header, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        body += b" " * ((-len(body)) % 8)   # deterministic space padding, block %8==0
        tmp = self.output_path.parent / f"{self.output_path.name}.tmp.{uuid.uuid4().hex}"
        with open(tmp, "wb") as f:
            f.write(pack_u64(len(body)))
            f.write(body)
            for n, rec, b, e, cache in plan:
                with open(cache, "rb") as src:
                    remaining = e - b
                    while remaining:
                        chunk = src.read(min(remaining, max(self.args.chunk_bytes, 64)))
                        if not chunk:
                            raise PrepareError(f"internal: truncated cache for {n!r}")
                        f.write(chunk)
                        remaining -= len(chunk)
            f.flush()
            os.fsync(f.fileno())
        osha = sha256_file(tmp)
        verify_output_file(tmp, self.output_path.name)
        if self.output_path.exists():
            # Only the exact same re-packaged content may land here (e.g. a
            # rerun after a crash between rename and manifest-complete).
            cur = sha256_file(self.output_path)
            if cur != osha:
                os.remove(tmp)
                raise PrepareError(
                    f"{self.output_path} exists with hash {cur}, this run produced "
                    f"{osha}: inconsistent existing output -- refusing to overwrite")
            os.remove(tmp)
        else:
            os.replace(tmp, self.output_path)
        manifest["output"] = {
            "path": self.args.output_name,
            "sha256": osha,
            "file_bytes": self.output_path.stat().st_size,
            "tensor_bytes": cursor,
            "header_block_bytes": 8 + len(body),
            "tensor_count": len(names),
        }
        manifest["status"] = "complete"
        self._save_manifest(manifest)
        self.http.log(f" -- wrote {self.output_path.name}: {len(names)} tensors, "
                      f"{cursor} bytes, sha256 {osha[:12]}...")

    # ---------- whole run ----------
    def _load_state(self):
        out = self.out_dir
        if out.exists() and any(out.iterdir()):
            manifest = self._load_existing_manifest()
            if manifest is None:
                raise PrepareError(
                    f"--output-dir {out} exists with content but no "
                    f"{MANIFEST_NAME}: refusing to overwrite files this tool cannot "
                    f"prove it wrote. Choose a fresh directory.")
            return manifest, False
        out.mkdir(parents=True, exist_ok=True)
        manifest = self._init_manifest()
        # Persist identity immediately: an interrupted FIRST run must remain
        # resumable (dir content without a manifest would otherwise be refused
        # as foreign), and the manifest is the only provenance anchor.
        self._save_manifest(manifest)
        return manifest, True

    def verify_complete(self, manifest: dict) -> None:
        """Rerun fast-path: every recorded artifact must still hash right;
        any divergence aborts instead of silently re-deriving provenance."""
        rec = manifest.get("output", {})
        if rec.get("path") != self.args.output_name:
            raise PrepareError("manifest output name disagrees with --output-name")
        if not self.output_path.exists():
            raise PrepareError(f"manifest is complete but {self.output_path} is "
                               f"missing/moved -- refusing to silently rebuild")
        actual = sha256_file(self.output_path)
        if actual != rec.get("sha256"):
            raise PrepareError(
                f"output hash {actual} != completed manifest hash "
                f"{rec.get('sha256')}: tampered output at a pinned revision; refusing "
                f"silent overwrite. Delete the directory manually after review, or "
                f"use a fresh one.")
        if rec.get("file_bytes") != self.output_path.stat().st_size:
            raise PrepareError("output size disagrees with the completed manifest")
        verify_output_file(self.output_path, self.args.output_name)
        for key in ("config", "index"):
            r = manifest.get(key, {})
            if not r.get("sha256"):
                raise PrepareError(f"completed manifest has no {key} record")
            saved = self.out_dir / Path(r["path"]).name
            if not saved.exists() or sha256_file(saved) != r["sha256"]:
                raise PrepareError(f"saved {r.get('path')} no longer matches the "
                                   f"manifest hash -- refusing")
        for name, t in manifest.get("tensors", {}).items():
            cache = self.tensors_dir / (self._cache_key(t["shard"], name) + ".bin")
            if not cache.exists() or cache.stat().st_size != t["bytes"] \
                    or sha256_file(cache) != t["sha256"]:
                raise PrepareError(f"cache diverged from completed manifest for "
                                   f"{name!r} -- refusing")
        if len(manifest.get("tensors", {})) != manifest["index"]["selected_tensors"]:
            raise PrepareError("tensor count disagrees with the index selection")

    def prepare(self) -> int:
        manifest, fresh = self._load_state()
        if not fresh and manifest.get("status") == "complete":
            self.verify_complete(manifest)
            self.http.log(" -- pinned manifest already proves this output complete: "
                          "nothing fetched (idempotent rerun)")
            print(f"OK already-complete {self.output_path} "
                  f"sha256={manifest['output']['sha256']}")
            return 0
        for d in (self.cache_dir, self.meta_dir, self.headers_dir, self.tensors_dir):
            d.mkdir(parents=True, exist_ok=True)
        sel, headers = self.build_plan(manifest)
        todo = []
        for name, shard in sorted(sel.selection.items()):
            rec = manifest["tensors"][name]
            if rec.get("sha256"):
                cache = self.tensors_dir / (self._cache_key(shard, name) + ".bin")
                if cache.exists() and cache.stat().st_size == rec["bytes"] \
                        and sha256_file(cache) == rec["sha256"]:
                    continue
            todo.append((name, shard))
        if todo:
            self.http.log(f" -- {len(todo)} tensor(s) to fetch, "
                          f"{len(sel.selection) - len(todo)} already proven-cached "
                          f"(jobs={self.args.jobs})")
        if self.args.jobs == 1:
            for name, shard in todo:
                self.fetch_tensor(manifest, name, shard, headers[shard])
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.jobs) as ex:
                futures = [ex.submit(self.fetch_tensor, manifest, name, shard,
                                     headers[shard]) for name, shard in todo]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
        if any(not rec.get("sha256") for rec in manifest["tensors"].values()):
            raise PrepareError("internal: tensors without provenance hash at "
                               "packaging time")
        self.build_output(manifest, sel)
        print(f"OK prepared {self.output_path} sha256={manifest['output']['sha256']} "
              f"tensors={manifest['output']['tensor_count']} "
              f"bytes={manifest['output']['tensor_bytes']}")
        return 0


def run_list(args) -> int:
    http = HttpSource(args.base_url, args.repo, args.revision, timeout=args.timeout,
                      retries=args.retries, backoff_max=args.backoff_max,
                      chunk_bytes=args.chunk_bytes, token=args.hf_token,
                      quiet=args.quiet)
    with tempfile.TemporaryDirectory(prefix="prepare_mtp_list_") as td:
        local = Path(td) / "index.json"
        http.fetch(local, args.index_file, what=f"index ({args.index_file})")
        index_obj = json.loads(local.read_text(encoding="utf-8"))
    sel = Selector(index_obj, args.tensor_prefix)
    print(f"repo={args.repo} revision={args.revision} prefix={args.tensor_prefix!r}")
    print(f"selected tensors: {len(sel.selection)} across {len(sel.shards)} shard(s)")
    for shard in sorted(sel.shards):
        for name in sorted(sel.shards[shard]):
            print(f"  {shard}  {name}")
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        prog="prepare_mtp_source.py",
        description="Fetch only the official pinned mtp.* tensors via HTTP Range "
                    "requests and repackage them unchanged into an MTP-only "
                    "safetensors, with a provenance manifest (stdlib only).")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--revision", required=True,
                   help="REQUIRED explicit full 40-hex commit SHA (no branch names)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="https in production; plain http only for loopback test servers")
    p.add_argument("--tensor-prefix", default=DEFAULT_PREFIX,
                   help="dot-boundary-exact prefix for selection (default: mtp.)")
    p.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    p.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    p.add_argument("--output-dir",
                   help="fresh, empty, or proven-compatible resume directory")
    p.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME,
                   help="basename of the MTP-only safetensors inside --output-dir")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="optional read token (or env HF_TOKEN); never recorded")
    p.add_argument("--jobs", type=int, default=1,
                   help="tensor fetch concurrency (sequential default; clamped 1..8)")
    p.add_argument("--retries", type=int, default=6,
                   help="attempts per HTTP fetch (transient failures only)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="per-operation socket timeout, seconds")
    p.add_argument("--backoff-max", type=float, default=30.0)
    p.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024,
                   help="streaming chunk size (bounds RAM during fetch and copy)")
    p.add_argument("--header-limit-bytes", type=int, default=32 * 1024 * 1024,
                   help="sane upper bound on a shard header JSON length")
    p.add_argument("--list", action="store_true",
                   help="inspection only: fetch the index, print the selection, "
                        "write nothing into --output-dir")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    args.revision = validate_revision(args.revision)
    validate_base_url(args.base_url)
    if not args.tensor_prefix:
        p.error("--tensor-prefix must be non-empty")
    matches_prefix("probe", args.tensor_prefix)   # rejects degenerate prefixes
    if not args.list:
        if not args.output_dir:
            p.error("--output-dir is required unless --list is given")
        args.jobs = max(1, min(int(args.jobs), 8))
    return args


def main(argv=None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except PrepareError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 1
    try:
        if args.list:
            return run_list(args)
        return MtpPreparer(args).prepare()
    except PrepareError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR interrupted (rerun with the same arguments to resume)",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
