#!/usr/bin/env python3
"""CPU-only tests for the two portable publication helpers.

Fakes: a temporary-directory fake sysfs tree, real LOCAL AF_UNIX sockets,
duck-typed torch/generator. NOTHING HERE TOUCHES A GPU, real sysfs, sudo or the
network; they prove the server's wire protocol, its restore guarantees and
filesystem refusals, and the summariser's published arithmetic.

Run from the repo root:
    PYTHONPYCACHEPREFIX=/tmp/rocm-publication-pycache python3 -m pytest -q \
        -p no:cacheprovider rocm_tools/rdna2/tests/test_publication_helpers_cpu.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2 import power_policy, power_server, summarize_tp

BDF_A = "0000:43:00.0"
BDF_B = "0000:03:00.0"
ATTR = power_server.POWER_LEVEL_ATTR
EVENT_FIELDS = {"label", "mode", "before", "after", "requested_unix_s",
                "per_gpu_write_ms", "both_write_ms", "write_and_readback_ms"}
JOIN_TIMEOUT = 20


def make_fake_sysfs(base, bdfs=(BDF_A, BDF_B), initial="auto"):
    """`<root>/<BDF>/power_dpm_force_performance_level`, exactly like Linux sysfs."""
    root = Path(base) / "sysfs"
    paths = []
    for bdf in bdfs:
        attr = root / bdf / ATTR
        attr.parent.mkdir(parents=True)
        attr.write_text(initial + "\n")
        paths.append(attr)
    return root, paths


def connect_with_retry(path, timeout=10.0):
    deadline = time.monotonic() + timeout
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(path))
            return sock
        except OSError:
            sock.close()
            if time.monotonic() > deadline:
                raise AssertionError(f"server socket {path} never became connectable")
            time.sleep(0.01)


def wait_for_socket(path, timeout=10.0):
    """Block until the server has bound `path` (a backlog connect would steal the
    single accept, so probe the inode, not the connection)."""
    deadline = time.monotonic() + timeout
    while not Path(path).is_socket():
        if time.monotonic() > deadline:
            raise AssertionError(f"server never bound {path}")
        time.sleep(0.01)


def read_modes(paths):
    return [Path(p).read_text().strip() for p in paths]


def rpc(sock, request, timeout=10.0):
    """One JSON-line request/reply on a raw socket, leaving the socket OPEN.

    No makefile and no close: the helper exits (and restores) the moment its
    client disconnects, so a test that wants to observe a signal must not end
    the session first.
    """
    sock.sendall((json.dumps(request) + "\n").encode())
    buf = b""
    deadline = time.monotonic() + timeout
    while not buf.endswith(b"\n"):
        try:
            sock.settimeout(max(0.1, deadline - time.monotonic()))
            chunk = sock.recv(4096)
        except OSError as e:
            raise AssertionError(f"no reply to {request!r}: {e!r}") from e
        if not chunk:
            raise AssertionError(f"server closed instead of replying to {request!r}")
        buf += chunk
    return json.loads(buf)


class RequestClient(threading.Thread):
    """One server client: sends JSON-line requests, records the JSON-line replies."""

    def __init__(self, path, requests, *, observe=None):
        super().__init__(daemon=True)
        self.path, self.requests = str(path), list(requests)
        self.observe = observe or (lambda sock, fh: None)
        self.replies, self.error, self.sock_stat = [], None, None

    def run(self):
        sock = None
        try:
            sock = connect_with_retry(self.path)
            with sock.makefile("rwb", buffering=0) as fh:
                self.observe(sock, fh)
                for req in self.requests:
                    fh.write((json.dumps(req) + "\n").encode())
                    reply = fh.readline()
                    self.replies.append(json.loads(reply) if reply else None)
        except BaseException as e:            # surfaced by the test, never swallowed
            self.error = e
        finally:
            if sock:
                sock.close()


def run_serve(paths, sock, report, requests, *, observe=None, **kwargs):
    """serve() in a worker thread with one client; returns (code, report, client)."""
    holder = {}

    def target():
        try:
            holder["result"] = power_server.serve(paths, sock, report, **kwargs)
        except BaseException as e:
            holder["raised"] = e

    server = threading.Thread(target=target)
    server.start()
    client = RequestClient(sock, requests, observe=observe)
    client.start()
    server.join(JOIN_TIMEOUT)
    client.join(JOIN_TIMEOUT)
    assert not server.is_alive(), "serve() never returned"
    assert not client.is_alive(), "client never finished"
    if "raised" in holder:
        raise holder["raised"]
    return holder["result"], client


# ===========================================================================
# power_server: validation and the internal switch API
# ===========================================================================

class BdfValidation(unittest.TestCase):
    def test_accepts_two_distinct_canonical_bdfs(self):
        self.assertEqual(power_server.validate_bdfs([BDF_A, BDF_B]), [BDF_A, BDF_B])

    def test_rejects_wrong_device_count(self):
        for values in ([], [BDF_A], [BDF_A, BDF_B, "0000:04:00.0"]):
            with self.subTest(values=values):
                with self.assertRaises(power_server.PowerServerError):
                    power_server.validate_bdfs(values)

    def test_rejects_non_canonical_forms(self):
        bad = ["43:00.0", "0000:43:00", "0000:43:00.00", "00000:43:00.0", "0000:430:00.0",
               "0000:43:00.00.0", "0000-43-00.0", "", " 0000:43:00.0", "0000:43:0.0 "]
        for bdf in bad:
            with self.subTest(bdf=bdf), self.assertRaises(power_server.PowerServerError):
                power_server.validate_bdfs([bdf, BDF_B])

    def test_rejects_uppercase_bdf(self):
        # sysfs names devices in lowercase; accepting both spellings would let the
        # same card be listed twice under two different paths.
        with self.assertRaises(power_server.PowerServerError):
            power_server.validate_bdfs(["0000:AB:00.0", BDF_B])

    def test_rejects_path_traversal_disguised_as_bdf(self):
        for bdf in ("../../etc", "0000:43:00.0/../../..", "..:..:..:."):
            with self.subTest(bdf=bdf), self.assertRaises(power_server.PowerServerError):
                power_server.validate_bdfs([bdf, BDF_B])

    def test_rejects_duplicates(self):
        with self.assertRaises(power_server.PowerServerError):
            power_server.validate_bdfs([BDF_A, BDF_A])

    def test_resolves_only_the_one_power_attribute(self):
        with tempfile.TemporaryDirectory() as td:
            root, paths = make_fake_sysfs(td)
            self.assertEqual(power_server.power_level_paths([BDF_A, BDF_B], root=root), paths)
            # Sibling attributes of the same device are NOT selectable.
            (root / BDF_A / "power_dpm_state").write_text("active\n")
            self.assertEqual(power_server.power_level_paths([BDF_A, BDF_B], root=root), paths)

    def test_missing_attribute_fails_before_any_switch(self):
        with tempfile.TemporaryDirectory() as td:
            root, _ = make_fake_sysfs(td, bdfs=(BDF_A,))
            with self.assertRaises(power_server.PowerServerError):
                power_server.power_level_paths([BDF_A, BDF_B], root=root)

    def test_cli_exposes_no_sysfs_root_or_attribute_override(self):
        opts = {o for action in power_server.build_parser()._actions
                for o in action.option_strings if o.startswith("--")}
        self.assertEqual(opts, {"--devices", "--help", "--report", "--socket"})


class InvokerOwnership(unittest.TestCase):
    def test_absent_sudo_env_means_no_handoff(self):
        self.assertIsNone(power_server.invoker_ids({}))
        self.assertIsNone(power_server.invoker_ids({"SUDO_UID": "1000"}))

    def test_parses_sudo_ids(self):
        self.assertEqual(power_server.invoker_ids({"SUDO_UID": "1000", "SUDO_GID": "100"}),
                         (1000, 100))
        self.assertEqual(power_server.invoker_ids({"SUDO_UID": "0", "SUDO_GID": "0"}), (0, 0))

    def test_rejects_unusable_sudo_ids(self):
        for env in ({"SUDO_UID": "root", "SUDO_GID": "1000"},
                    {"SUDO_UID": "-1", "SUDO_GID": "1000"},
                    {"SUDO_UID": "1000", "SUDO_GID": "-5"}):
            with self.subTest(env=env), self.assertRaises(power_server.PowerServerError):
                power_server.invoker_ids(env)


class SwitchMechanics(unittest.TestCase):
    def test_switches_both_devices_and_confirms_by_readback(self):
        with tempfile.TemporaryDirectory() as td:
            _root, paths = make_fake_sysfs(td)
            event = power_server.apply_mode(paths, "profile_peak", "label-x")
            self.assertEqual(set(event), EVENT_FIELDS)
            self.assertEqual(event["label"], "label-x")
            self.assertEqual(event["mode"], "profile_peak")
            self.assertEqual(event["before"], ["auto", "auto"])
            self.assertEqual(event["after"], ["profile_peak", "profile_peak"])
            self.assertEqual(len(event["per_gpu_write_ms"]), 2)
            self.assertTrue(all(v >= 0 for v in event["per_gpu_write_ms"]))
            self.assertLessEqual(event["both_write_ms"], event["write_and_readback_ms"])
            self.assertEqual(read_modes(paths), ["profile_peak", "profile_peak"])

    def test_requested_unix_s_is_wall_clock(self):
        with tempfile.TemporaryDirectory() as td:
            _root, paths = make_fake_sysfs(td)
            before = time.time()
            event = power_server.apply_mode(paths, "auto", "l")
            self.assertGreaterEqual(event["requested_unix_s"], before)
            self.assertLessEqual(event["requested_unix_s"], time.time())

    def test_invalid_mode_is_refused_without_touching_devices(self):
        with tempfile.TemporaryDirectory() as td:
            _root, paths = make_fake_sysfs(td)
            with self.assertRaises(power_server._ProtocolError):
                power_server.apply_mode(paths, "profile_max", "l")
            self.assertEqual(read_modes(paths), ["auto", "auto"])

    def test_silent_device_yields_protocol_error_with_partial_event(self):
        with tempfile.TemporaryDirectory() as td:
            _root, paths = make_fake_sysfs(td)
            # A device that accepts the write but never changes (real firmware failure).
            with mock.patch.object(power_server, "_write_mode", return_value=None):
                with self.assertRaises(power_server._ProtocolError) as caught:
                    power_server.apply_mode(paths, "profile_peak", "l")
            event = caught.exception.event
            self.assertIsNotNone(event)
            self.assertEqual(set(event), EVENT_FIELDS)
            self.assertEqual(event["after"], ["auto", "auto"])

    def test_restore_writes_the_captured_originals(self):
        with tempfile.TemporaryDirectory() as td:
            _root, paths = make_fake_sysfs(td)
            power_server.apply_mode(paths, "profile_peak", "l")
            final, errors = power_server.restore_modes(paths, ["auto", "auto"])
            self.assertEqual((final, errors), (["auto", "auto"], []))

    def test_restore_failures_are_collected_not_raised(self):
        with tempfile.TemporaryDirectory() as td:
            _root, paths = make_fake_sysfs(td)
            power_server.apply_mode(paths, "profile_peak", "l")
            with mock.patch.object(power_server, "_write_mode",
                                   side_effect=OSError("sysfs busy")):
                final, errors = power_server.restore_modes(paths, ["auto", "auto"])
            self.assertEqual(final, ["profile_peak", "profile_peak"])
            self.assertEqual(len(errors), 2)

    def test_restore_handles_asymmetric_originals(self):
        # A previous crashed run can leave one card in profile_peak: both are
        # restored to their OWN captured value, not to a shared assumption.
        with tempfile.TemporaryDirectory() as td:
            root, paths = make_fake_sysfs(td)
            paths[1].write_text("profile_peak\n")
            final, errors = power_server.restore_modes(paths, ["auto", "profile_peak"])
            self.assertEqual(final, ["auto", "profile_peak"])
            self.assertEqual(errors, [])
            self.assertEqual(read_modes(paths), ["auto", "profile_peak"])
            self.assertTrue((root / BDF_B / ATTR).exists())


# ===========================================================================
# power_server: session over a real local Unix socket
# ===========================================================================

class ServerSession(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        _root, self.paths = make_fake_sysfs(self.td.name)
        self.sock = Path(self.td.name) / "power.sock"
        self.report = Path(self.td.name) / "power.json"

    def serve(self, requests, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_serve(self.paths, self.sock, self.report, requests, **kwargs)

    def test_serves_two_requests_and_exits_clean(self):
        (code, report), client = self.serve([{"mode": "profile_peak", "label": "before-decode"},
                                             {"mode": "auto", "label": "after-drained"}])
        self.assertEqual(code, power_server.EXIT_CLEAN)
        self.assertIsNone(client.error, repr(client.error))
        self.assertEqual([r["mode"] for r in client.replies], ["profile_peak", "auto"])
        for reply in client.replies:
            self.assertEqual(set(reply), EVENT_FIELDS)
        self.assertEqual(client.replies[0]["after"], ["profile_peak", "profile_peak"])
        self.assertEqual(client.replies[0]["before"], ["auto", "auto"])
        self.assertEqual(client.replies[0]["label"], "before-decode")
        self.assertEqual(report["original"], ["auto", "auto"])
        self.assertEqual(report["final"], ["auto", "auto"])
        self.assertTrue(report["restored"])
        self.assertEqual([e["mode"] for e in report["events"]], ["profile_peak", "auto"])
        self.assertEqual(report["cleanup_errors"], [])
        self.assertNotIn("error", report)

    def test_exits_after_the_single_client_disconnects(self):
        (code, _report), _client = self.serve([{"mode": "auto", "label": "l"}])
        self.assertEqual(code, power_server.EXIT_CLEAN)
        self.assertFalse(self.sock.exists(), "socket must be cleaned up on exit")

    def test_report_is_durable_private_and_describes_the_session(self):
        (_code, _report), _client = self.serve([{"mode": "profile_peak", "label": "a"},
                                               {"mode": "auto", "label": "b"}])
        on_disk = json.loads(self.report.read_text())
        self.assertEqual([e["label"] for e in on_disk["events"]], ["a", "b"])
        self.assertEqual(on_disk["devices"], [str(p) for p in self.paths])
        self.assertEqual(on_disk["socket"], str(self.sock))
        self.assertEqual(on_disk["pid"], os.getpid())
        self.assertTrue(on_disk["restored"])
        self.assertEqual(self.report.stat().st_mode & 0o777, 0o600)
        self.assertLessEqual(on_disk["started_unix_s"], on_disk["finished_unix_s"])

    def test_socket_is_0600_and_handed_to_the_invoker(self):
        seen = {}

        def observe(sock, fh):
            stat = os.stat(self.sock)
            seen["mode"] = stat.st_mode & 0o777
            seen["ids"] = (stat.st_uid, stat.st_gid)

        owner = (os.getuid(), os.getgid())
        (code, report), _client = self.serve([{"mode": "profile_peak", "label": "up"}],
                                             observe=observe, owner=owner)
        self.assertEqual(code, power_server.EXIT_CLEAN)
        self.assertTrue(report["restored"])
        self.assertEqual(seen["mode"], 0o600)
        self.assertEqual(seen["ids"], owner)

    def test_socket_planted_at_our_path_during_session_survives(self):
        # Cleanup must remove only the inode it bound: a file someone else planted
        # at the same path mid-session is theirs, not ours to delete.
        planted = "not a socket\n"

        def observe(sock, fh):
            os.unlink(self.sock)
            self.sock.write_text(planted)

        (code, report), _client = self.serve([{"mode": "profile_peak", "label": "up"}],
                                             observe=observe)
        self.assertEqual(code, power_server.EXIT_CLEAN)
        self.assertEqual(self.sock.read_text(), planted)
        self.assertEqual(report["cleanup_errors"], [])
        self.assertTrue(report["restored"])
        self.assertEqual(read_modes(self.paths), ["auto", "auto"])

    def test_chown_targets_the_sudo_ids_when_launched_by_sudo(self):
        with tempfile.TemporaryDirectory() as td:
            _root, paths = make_fake_sysfs(td)
            sock, report = Path(td) / "p.sock", Path(td) / "p.json"
            calls = []
            real_chown = os.chown

            def record(path, uid, gid, *args, **kwargs):
                calls.append((os.path.basename(str(path)), uid, gid))
                return real_chown(path, os.getuid(), os.getgid(), *args, **kwargs)

            with mock.patch("os.chown", side_effect=record), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_serve(paths, sock, report, [{"mode": "profile_peak", "label": "up"}],
                          owner=(4242, 4343))
            self.assertIn(("p.sock", 4242, 4343), calls)
            self.assertIn(("p.json", 4242, 4343), calls)

    def test_malformed_request_is_a_session_error_and_restores(self):
        holder = {}

        def target():
            try:
                holder["result"] = power_server.serve(self.paths, self.sock, self.report)
            except BaseException as e:
                holder["raised"] = e

        def junk_client():
            sock = connect_with_retry(self.sock)
            sock.sendall(b"not json at all\n")
            sock.close()

        server, client = threading.Thread(target=target), threading.Thread(target=junk_client)
        server.start()
        client.start()
        with contextlib.redirect_stdout(io.StringIO()):
            server.join(JOIN_TIMEOUT)
            client.join(JOIN_TIMEOUT)
        self.assertFalse(server.is_alive())
        if "raised" in holder:
            raise holder["raised"]
        code, report = holder["result"]
        self.assertEqual(code, power_server.EXIT_SESSION)
        self.assertIn("not one JSON object", report["error"])
        self.assertEqual(read_modes(self.paths), ["auto", "auto"])
        self.assertTrue(report["restored"])

    def test_invalid_mode_restores_originals_and_reports_to_the_client(self):
        (code, report), client = self.serve([{"mode": "high", "label": "nope"}])
        self.assertEqual(code, power_server.EXIT_SESSION)
        self.assertIn("invalid mode", report["error"])
        self.assertEqual(read_modes(self.paths), ["auto", "auto"], "must be restored")
        self.assertTrue(report["restored"])
        self.assertEqual(report["events"], [], "a refused mode is not a switch")
        self.assertIn("error", client.replies[0])
        self.assertNotIn("after", client.replies[0])

    def test_unconfirmed_switch_is_recorded_restored_and_reported(self):
        with mock.patch.object(power_server, "_write_mode", return_value=None):
            (code, report), client = self.serve([{"mode": "profile_peak", "label": "l"}])
        self.assertEqual(code, power_server.EXIT_SESSION)
        self.assertIn("does not match requested", report["error"])
        self.assertEqual(len(report["events"]), 1)
        self.assertEqual(report["events"][0]["after"], ["auto", "auto"])
        self.assertEqual(read_modes(self.paths), ["auto", "auto"])
        self.assertTrue(report["restored"])
        self.assertIn("error", client.replies[0])
        self.assertEqual(client.replies[0]["after"], ["auto", "auto"])

    def test_preexisting_socket_is_refused_not_deleted(self):
        self.sock.write_text("keep me\n")
        with self.assertRaises(power_server.PowerServerError):
            power_server.serve(self.paths, self.sock, self.report)
        self.assertEqual(self.sock.read_text(), "keep me\n")
        self.assertFalse(self.report.exists(), "no artifact for a refused session")
        self.assertEqual(read_modes(self.paths), ["auto", "auto"])

    def test_symlinked_socket_is_refused(self):
        (Path(self.td.name) / "elsewhere.sock").symlink_to(self.sock)
        link = Path(self.td.name) / "elsewhere.sock"
        with self.assertRaises(power_server.PowerServerError):
            power_server.serve(self.paths, link, self.report)
        self.assertFalse(self.report.exists())

    def test_preexisting_report_is_never_overwritten(self):
        self.report.write_text("precious\n")
        with self.assertRaises(power_server.PowerServerError):
            power_server.serve(self.paths, self.sock, self.report)
        self.assertEqual(self.report.read_text(), "precious\n")
        self.assertFalse(self.sock.exists(), "must refuse before binding")

    def test_symlinked_report_is_not_followed(self):
        victim = Path(self.td.name) / "victim.json"
        victim.write_text("do not touch\n")
        self.report.symlink_to(victim)
        with self.assertRaises(power_server.PowerServerError):
            power_server.serve(self.paths, self.sock, self.report)
        self.assertEqual(victim.read_text(), "do not touch\n")
        self.assertFalse(self.sock.exists())

    def test_restore_failure_is_reported_as_not_restored(self):
        def flaky(path, mode):
            if mode == "auto":
                raise OSError("device refuses to come back")
            Path(path).write_text(mode + "\n")

        with mock.patch.object(power_server, "_write_mode", side_effect=flaky):
            (code, report), _client = self.serve([{"mode": "profile_peak", "label": "up"}])
        self.assertEqual(code, power_server.EXIT_CLEAN)   # the session itself was clean
        self.assertFalse(report["restored"])
        self.assertEqual(len(report["restore_errors"]), 2)
        self.assertEqual(report["final"], ["profile_peak", "profile_peak"])
        self.assertEqual(read_modes(self.paths), ["profile_peak", "profile_peak"])

    def test_signal_while_restoring_does_not_abort_the_restore(self):
        self.addCleanup(power_server._RESTORING.update, {"active": False})
        with self.assertRaises(power_server._Interrupted):
            power_server.interrupt_handler(signal.SIGTERM, None)
        power_server._RESTORING["active"] = True
        self.assertIsNone(power_server.interrupt_handler(signal.SIGTERM, None))


class ServerInterruptSubprocess(unittest.TestCase):
    """SIGINT/SIGTERM must reach a server blocked in accept(), in its own process."""

    SCRIPT = (
        "import sys\n"
        "from pathlib import Path\n"
        "from rocm_tools.rdna2 import power_server as ps\n"
        "ps.install_signal_handlers()\n"
        "code, report = ps.serve([Path(p) for p in sys.argv[1:3]], sys.argv[3], sys.argv[4])\n"
        "sys.exit(code)\n")

    def run_real_process(self, sig):
        """Drive a REAL child process: switch to peak, hold the connection open so
        the server sits in readline(), then signal it and prove it restored."""
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, True)
        _root, paths = make_fake_sysfs(td)
        sock, report = Path(td) / "p.sock", Path(td) / "p.json"
        repo = str(Path(__file__).resolve().parents[3])
        proc = subprocess.Popen([sys.executable, "-c", self.SCRIPT,
                                 str(paths[0]), str(paths[1]), str(sock), str(report)],
                                cwd=repo,
                                env={**os.environ, "PYTHONPATH": repo,
                                     "PYTHONPYCACHEPREFIX": os.environ.get(
                                         "PYTHONPYCACHEPREFIX", "/tmp/rocm-publication-pycache")},
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        wait_for_socket(sock)
        # Raw sendall/recv, and NO close before signalling: a makefile() context
        # manager would close the socket, and the helper (correctly) restores on
        # disconnect -- which would race the signal we are trying to test. The server
        # must still be blocked in readline() when the signal arrives.
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(sock))
        try:
            reply = rpc(client, {"mode": "profile_peak", "label": "hold"})
            self.assertEqual(reply["after"], ["profile_peak", "profile_peak"])
            self.assertEqual(read_modes(paths), ["profile_peak", "profile_peak"],
                             "the peak switch must be live before signalling")
            proc.send_signal(sig)                 # server is blocked in readline()
            try:
                code = proc.wait(timeout=JOIN_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.fail(f"server ignored {sig.name} while blocked in readline()")
        finally:
            client.close()
        return code, json.loads(report.read_text()), read_modes(paths), sock

    def test_sigterm_restores_the_original_modes(self):
        code, data, modes, sock = self.run_real_process(signal.SIGTERM)
        self.assertEqual(code, 128 + signal.SIGTERM)
        self.assertEqual(modes, ["auto", "auto"])
        self.assertEqual(data["interrupted"], "SIGTERM")
        self.assertTrue(data["restored"])
        self.assertEqual(data["restore_errors"], [])
        self.assertFalse(sock.exists())
        self.assertEqual([e["mode"] for e in data["events"]], ["profile_peak"])

    def test_sigint_restores_the_original_modes(self):
        code, data, modes, sock = self.run_real_process(signal.SIGINT)
        self.assertEqual(code, 128 + signal.SIGINT)
        self.assertEqual(modes, ["auto", "auto"])
        self.assertEqual(data["interrupted"], "SIGINT")
        self.assertTrue(data["restored"])
        self.assertFalse(sock.exists())


class PowerPolicyInterop(unittest.TestCase):
    """The published server must satisfy the published client, protocol unchanged."""

    class FakeTorch:
        def __init__(self):
            self.cuda = SimpleNamespace(synchronize=lambda dev: None)

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        _root, self.paths = make_fake_sysfs(self.td.name)
        self.sock = Path(self.td.name) / "power.sock"
        self.report = Path(self.td.name) / "power.json"
        self.gen = SimpleNamespace(draft_model=None, ngram_match_min=0, active_jobs=[])
        self.holder = {}

    def start_server(self):
        def target():
            try:
                self.holder["result"] = power_server.serve(self.paths, self.sock, self.report)
            except BaseException as e:
                self.holder["raised"] = e

        self.server = threading.Thread(target=target)
        self.server.start()
        wait_for_socket(self.sock)
        self.addCleanup(lambda: self.server.join(JOIN_TIMEOUT))

    def test_batch_gt1_adapter_holds_peak_then_returns_to_auto(self):
        self.start_server()
        adapter = power_policy.attach(self.gen, self.FakeTorch(), 2, [0, 1], self.sock)
        with adapter:
            self.assertEqual(adapter.applied, "profile_peak")
            self.assertEqual(read_modes(self.paths), ["profile_peak", "profile_peak"])
        self.server.join(JOIN_TIMEOUT)
        self.assertFalse(self.server.is_alive(), "the adapter's close must end the session")
        if "raised" in self.holder:
            raise self.holder["raised"]
        code, report = self.holder["result"]
        self.assertEqual(code, power_server.EXIT_CLEAN)
        self.assertEqual([e["mode"] for e in report["events"]], ["profile_peak", "auto"])
        self.assertTrue(report["restored"])
        self.assertEqual(read_modes(self.paths), ["auto", "auto"])
        # The client verified OUR readbacks rather than trusting its own request.
        self.assertEqual([r["after"] for r in adapter.records],
                         [["profile_peak", "profile_peak"], ["auto", "auto"]])
        self.assertFalse(adapter.summary()["helper_unverified"])
        self.assertTrue(all(r["rpc_ms"] >= 0 for r in adapter.records))

    def test_adapter_fails_closed_when_readback_does_not_confirm(self):
        self.start_server()
        adapter = power_policy.attach(self.gen, self.FakeTorch(), 2, [0, 1], self.sock)
        with mock.patch.object(power_server, "_write_mode", return_value=None):
            with self.assertRaises(power_policy.PowerPolicyError):
                with adapter:
                    pass
        self.server.join(JOIN_TIMEOUT)
        _code, report = self.holder["result"]
        self.assertTrue(report["restored"])
        self.assertEqual(read_modes(self.paths), ["auto", "auto"])


class ServerCli(unittest.TestCase):
    # main() installs signal handlers, so it must run on the main thread; the
    # client is the thread here.

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root, self.paths = make_fake_sysfs(self.td.name)
        self.sock = Path(self.td.name) / "power.sock"
        self.report = Path(self.td.name) / "power.json"
        self.argv = ["--devices", BDF_A, BDF_B, "--socket", str(self.sock),
                     "--report", str(self.report)]

    def drive(self, requests, *, argv=None, root=None, **patches):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(power_server, "SYSFS_PCI_ROOT",
                                              str(self.root if root is None else root)))
        for name, kwargs in patches.items():
            stack.enter_context(mock.patch.object(power_server, name, **kwargs))
        if requests:
            path = self.sock if argv is None else Path(argv[argv.index("--socket") + 1])
            client = RequestClient(path, requests)
            client.start()
        else:                                   # refusal expected: nothing may connect
            client = None
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = power_server.main(self.argv if argv is None else argv)
        if client:
            client.join(JOIN_TIMEOUT)
            self.assertFalse(client.is_alive())
        return code, client, out.getvalue(), err.getvalue()

    def test_clean_run_exits_zero(self):
        code, client, out, err = self.drive([{"mode": "profile_peak", "label": "up"},
                                             {"mode": "auto", "label": "down"}])
        self.assertEqual(code, 0)
        self.assertEqual(read_modes(self.paths), ["auto", "auto"])
        self.assertIsNone(client.error, repr(client.error))
        self.assertIn("READY", out)
        self.assertIn("RESTORED True", out)
        self.assertEqual(err, "")

    def test_not_restored_run_exits_one(self):
        def flaky(path, mode):
            if mode == "auto":
                raise OSError("refused")
            Path(path).write_text(mode + "\n")

        code, _client, _out, err = self.drive([{"mode": "profile_peak", "label": "up"}],
                                              _write_mode={"side_effect": flaky})
        self.assertEqual(code, power_server.EXIT_NOT_RESTORED)
        self.assertIn("NOT restored", err)

    def test_duplicate_bdf_is_refused_before_touching_sysfs(self):
        argv = ["--devices", BDF_A, BDF_A, "--socket", str(self.sock),
                "--report", str(self.report)]
        code, _client, _out, err = self.drive([], argv=argv)
        self.assertEqual(code, power_server.EXIT_SETUP)
        self.assertIn("distinct", err)
        self.assertFalse(self.report.exists())

    def test_absent_device_is_refused(self):
        empty = Path(self.td.name) / "empty-sysfs"
        empty.mkdir()
        code, _client, _out, err = self.drive([], root=empty)
        self.assertEqual(code, power_server.EXIT_SETUP)
        self.assertIn("power-level attribute not present", err)

    def test_preexisting_socket_is_refused_by_the_cli(self):
        self.sock.write_text("other endpoint\n")
        code, _client, _out, err = self.drive([])
        self.assertEqual(code, power_server.EXIT_SETUP)
        self.assertIn("already exists", err)
        self.assertEqual(self.sock.read_text(), "other endpoint\n")

    def test_creates_a_private_output_directory(self):
        deep = Path(self.td.name) / "runs" / "session-1"
        argv = ["--devices", BDF_A, BDF_B, "--socket", str(deep / "p.sock"),
                "--report", str(deep / "p.json")]
        code, _client, _out, err = self.drive([{"mode": "auto", "label": "l"}], argv=argv)
        self.assertEqual(code, 0, err)
        self.assertEqual((deep / "p.json").stat().st_mode & 0o777, 0o600)
        self.assertTrue(deep.is_dir())

    def test_cli_accepts_only_two_devices(self):
        argv = ["--devices", BDF_A, "--socket", str(self.sock), "--report", str(self.report)]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                power_server.main(argv)

    def test_help_has_no_side_effects_and_needs_nothing_privileged(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as caught:
                power_server.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        text = out.getvalue()
        self.assertIn("--devices", text)
        self.assertIn("power_dpm_force_performance_level", text)
        self.assertFalse(self.sock.exists())
        self.assertFalse(self.report.exists())


# ===========================================================================
# summarize_tp: published arithmetic on synthetic timing
# ===========================================================================

def row(ids, prompt_tokens, new_tokens, time_prefill, time_generate, events,
        *, language="ja", repeat=0, accepted=None, rejected=None):
    data = {"ids_sha256": ids, "language": language, "repeat": repeat,
            "prompt_tokens": prompt_tokens, "new_tokens": new_tokens,
            "time_prefill": time_prefill, "time_generate": time_generate,
            "delivery_events": [list(e) for e in events],
            "first_delivery_s": events[0][0]}
    if accepted is not None:
        data["accepted_draft_tokens"] = accepted
    if rejected is not None:
        data["rejected_draft_tokens"] = rejected
    return data


def report_of(groups, *, complete=True, capacity_only=False, validation_only=False,
              audit=True):
    """groups: [(timed, rows, wall_s), ...] flattened exactly like tp_run does.

    Rows are deep-copied so a test that corrupts one report cannot corrupt the
    shared module-level fixtures.
    """
    runs, entries = [], []
    for gi, (timed, rows_, wall_s) in enumerate(groups):
        fresh = [copy.deepcopy(r) for r in rows_]
        runs.extend(fresh)
        entries.append({"group": gi, "timed": timed, "jobs": len(fresh), "wall_s": wall_s,
                        "ids_sha256": [r["ids_sha256"] for r in fresh]})
    data = {"complete": complete, "capacity_only": capacity_only,
            "validation_only": validation_only, "groups": entries, "runs": runs}
    if audit:
        data["tp_final_audit"] = {"ranks": [{"device": 0, "torch_peak_bytes": 2 ** 30},
                                            {"device": 1, "torch_peak_bytes": 3 * 2 ** 29}]}
    return data


# Hand-computed fixtures (units: seconds, tokens, tokens/s).
# Group 0 -- batch 2, wall 4.0 s:
#   row1 [[1.0,5],[2.0,12],[3.0,20]]  row2 [[1.5,4],[2.5,10],[4.0,18]]
#   common window 1.5..3.0 s: (20+10)-(5+4) = 21 tokens / 1.5 s        = 14.0
#   full span     1.0..4.0 s: (20+18)-5     = 33 tokens / 3.0 s        = 11.0 (5.5/seq)
#   conservative prefill (10+12)/max(1.0,1.5) = 22/1.5;  e2e 38/4.0 = 9.5
# Group 1 -- batch 1, wall 2.0 s: [[0.5,2],[1.5,12]] -> 10 tokens / 1.0 s = 10.0,
#   full span identical, prefill 8/0.5 = 16.0, e2e 12/2.0 = 6.0.
GROUP0 = [row("a" * 8, 10, 20, 1.0, 2.0, [[1.0, 5], [2.0, 12], [3.0, 20]],
              accepted=6, rejected=4),
          row("b" * 8, 12, 18, 1.5, 2.5, [[1.5, 4], [2.5, 10], [4.0, 18]], accepted=3)]
GROUP1 = [row("c" * 8, 8, 12, 0.5, 1.0, [[0.5, 2], [1.5, 12]], repeat=1,
              accepted=0, rejected=0)]


class SummarizeArithmetic(unittest.TestCase):
    def setUp(self):
        self.data = report_of([(True, GROUP0, 4.0), (True, GROUP1, 2.0)])
        self.out = summarize_tp.summarize(self.data, name="r.json")

    def test_group_shape_and_row_binding(self):
        self.assertEqual([g["group"] for g in self.out["groups"]], [0, 1])
        self.assertEqual(self.out["report"], "r.json")
        self.assertEqual(self.out["groups"][0]["prompt_tokens_each"], [10, 12])
        self.assertEqual(self.out["groups"][0]["accepted"], 9)   # 6 + 3
        self.assertEqual(self.out["groups"][0]["rejected"], 4)   # 4 + missing -> 0
        self.assertEqual(self.out["groups"][1]["language"], "ja")
        self.assertEqual(self.out["groups"][1]["repeat"], 1)
        self.assertEqual(self.out["groups"][0]["metrics"]["run_indices"], [0, 1])
        self.assertEqual(self.out["groups"][1]["metrics"]["run_indices"], [2])

    def test_full_decode_span_includes_drain_and_excludes_prefill(self):
        span = self.out["groups"][0]["metrics"]["full_decode_span"]
        self.assertEqual((span["start_s"], span["end_s"], span["duration_s"]),
                         (1.0, 4.0, 3.0))
        self.assertEqual(span["token_delta"], 33)
        self.assertEqual(span["aggregate_tps"], 11.0)
        self.assertEqual(span["per_sequence_average_tps"], 5.5)
        self.assertIn("queue-drain", span["note"])
        self.assertIn("first delivery burst", span["note"])

    def test_common_window_comes_from_the_core_aggregator(self):
        metrics = self.out["groups"][0]["metrics"]
        window = metrics["common_window"]
        self.assertEqual((window["start_s"], window["end_s"], window["duration_s"]),
                         (1.5, 3.0, 1.5))
        self.assertEqual(window["delivered_at_start"], 9)
        self.assertEqual(window["delivered_at_end"], 30)
        self.assertEqual(window["token_delta"], 21)
        self.assertEqual(window["aggregate_decode_tps"], 14.0)
        self.assertEqual(metrics["jobs"], 2)
        self.assertEqual(metrics["input_tokens_total"], 22)
        self.assertEqual(metrics["prefill_makespan_s_engine"], 1.5)
        self.assertEqual(metrics["ttft_engine_s"], {"min": 1.0, "max": 1.5, "jobs": 2})
        self.assertEqual(metrics["end_to_end"]["aggregate_tps"], 9.5)
        self.assertEqual(metrics["overlap_note"], None)

    def test_conservative_prefill_divides_by_the_last_first_delivery(self):
        self.assertAlmostEqual(
            self.out["groups"][0]["aggregate_prefill_to_last_first_delivery_tps"], 22 / 1.5)
        self.assertAlmostEqual(
            self.out["groups"][1]["aggregate_prefill_to_last_first_delivery_tps"], 16.0)

    def test_batch1_equivalence_is_structural(self):
        one = self.out["groups"][1]["metrics"]
        self.assertEqual(one["common_window"]["aggregate_decode_tps"], 10.0)
        self.assertEqual(one["full_decode_span"]["aggregate_tps"],
                         one["common_window"]["aggregate_decode_tps"])
        self.assertEqual(one["per_job_decode_tps"][0]["observed_tps"], 10.0)
        self.assertEqual(one["per_job_decode_tps"][0]["engine_tps"], (12 - 1) / 1.0)

    def test_medians_ranges_and_acceptance_per_language(self):
        ja = self.out["languages"]["ja"]
        self.assertEqual(ja["groups"], 2)
        self.assertAlmostEqual(ja["acceptance"], 9 / 13)
        expected = {"full_span_aggregate_decode_tps": (10.5, 10.0, 11.0),
                    "full_span_per_sequence_decode_tps": (7.75, 5.5, 10.0),
                    "aggregate_decode_tps": (12.0, 10.0, 14.0),
                    "per_sequence_common_decode_tps": (8.5, 7.0, 10.0),
                    "aggregate_prefill_tps": ((16.0 + 22 / 1.5) / 2, 22 / 1.5, 16.0),
                    "last_ttft_s": (1.0, 0.5, 1.5),
                    "end_to_end_tps": (7.75, 6.0, 9.5)}
        self.assertEqual(set(ja), {"groups", "acceptance"} | set(expected))
        for key, (median, lo, hi) in expected.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(ja[key]["median"], median, places=9)
                self.assertAlmostEqual(ja[key]["min"], lo, places=9)
                self.assertAlmostEqual(ja[key]["max"], hi, places=9)

    def test_acceptance_without_draft_tokens_is_none_not_zero(self):
        data = report_of([(True, [row("d" * 8, 4, 5, 0.5, 1.0, [[0.5, 1], [1.5, 5]])], 2.0)])
        self.assertIsNone(summarize_tp.summarize(data)["languages"]["ja"]["acceptance"])

    def test_peak_memory_from_the_final_audit(self):
        self.assertEqual(self.out["peak_GiB"], {"0": 1.0, "1": 1.5})

    def test_untimed_groups_are_kept_but_never_summarised(self):
        data = report_of([(True, GROUP0, 4.0), (False, GROUP1, 2.0)])
        out = summarize_tp.summarize(data)
        self.assertEqual(out["languages"]["ja"]["groups"], 1)
        self.assertEqual(len(out["groups"]), 2)
        self.assertEqual(out["groups"][1]["timed"], False)

    def test_second_language_is_aggregated_separately(self):
        en = [row("e" * 8, 6, 6, 1.0, 1.0, [[1.0, 2], [2.0, 6]], language="en")]
        data = report_of([(True, GROUP1, 2.0), (True, en, 2.0)])
        out = summarize_tp.summarize(data)
        self.assertEqual(sorted(out["languages"]), ["en", "ja"])
        self.assertEqual(out["languages"]["en"]["groups"], 1)


class SummarizeRejections(unittest.TestCase):
    def base(self):
        return report_of([(True, GROUP0, 4.0)])

    def assert_rejects(self, data, needle=None):
        with self.assertRaises(summarize_tp.ReportError) as caught:
            summarize_tp.summarize(data, name="r.json")
        if needle:
            self.assertIn(needle, str(caught.exception))

    def test_incomplete_report(self):
        data = self.base()
        data["complete"] = False
        self.assert_rejects(data, "not complete")

    def test_capacity_only_report(self):
        data = self.base()
        data["capacity_only"] = True
        self.assert_rejects(data, "capacity_only")

    def test_validation_only_report(self):
        data = self.base()
        data["validation_only"] = True
        self.assert_rejects(data, "validation_only")

    def test_missing_or_empty_sections(self):
        no_runs = self.base()
        no_runs.pop("runs")
        empty_groups = self.base()
        empty_groups["groups"] = []
        bad_groups = self.base()
        bad_groups["groups"] = "nope"
        for label, arg in (("no runs", no_runs), ("empty groups", empty_groups),
                           ("groups not a list", bad_groups), ("top level is a list", ["x"])):
            with self.subTest(case=label):
                with self.assertRaises(summarize_tp.ReportError):
                    summarize_tp.summarize(arg, name="r.json")

    def test_group_row_hash_mismatch(self):
        data = self.base()
        data["runs"][0]["ids_sha256"] = "ff" * 8
        self.assert_rejects(data, "ids_sha256")

    def test_group_claiming_more_rows_than_exist(self):
        data = self.base()
        data["groups"][0]["jobs"] = 3
        self.assert_rejects(data, "remain at index")

    def test_trailing_run_covered_by_no_group(self):
        data = report_of([(True, GROUP0, 4.0), (True, GROUP1, 2.0)])
        data["groups"].pop()
        self.assert_rejects(data, "internally inconsistent")

    def test_missing_required_row_field(self):
        for key in ("delivery_events", "first_delivery_s", "prompt_tokens", "language"):
            data = self.base()
            del data["runs"][0][key]
            with self.subTest(key=key), self.assertRaises(summarize_tp.ReportError) as caught:
                summarize_tp.summarize(data, name="r.json")
            self.assertIn(key, str(caught.exception))

    def test_zero_length_full_span_is_rejected_not_infinite(self):
        flat = [row("f" * 8, 4, 4, 0.5, 0.5, [[1.0, 4], [1.0, 4]])]
        self.assert_rejects(report_of([(True, flat, 1.0)]), "decode span")

    def test_unusable_common_window_is_rejected_not_imputed(self):
        # The second job delivers only AFTER the first finished: no common overlap,
        # so the core aggregator emits nulls and summarising would be fiction.
        early = row("g" * 8, 4, 3, 0.5, 0.5, [[0.5, 1], [1.0, 3]])
        late = row("h" * 8, 4, 3, 0.5, 0.5, [[2.0, 1], [3.0, 3]])
        self.assert_rejects(report_of([(True, [early, late], 4.0)]), "null")

    def test_language_without_timed_groups_has_no_entry(self):
        self.assertEqual(summarize_tp.summarize(report_of([(False, GROUP0, 4.0)]))["languages"],
                         {})


class SummarizeCli(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)

    def write(self, name, data):
        """Write a report file: a dict is JSON-encoded, a str goes in verbatim."""
        path = Path(self.td.name) / name
        path.write_text(data if isinstance(data, str) else json.dumps(data))
        return path

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = summarize_tp.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_writes_sibling_summary_and_concise_stdout(self):
        good = self.write("run-a.json", report_of([(True, GROUP0, 4.0), (True, GROUP1, 2.0)]))
        code, out, err = self.run_main([str(good)])
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        summary = json.loads((Path(self.td.name) / "run-a-summary.json").read_text())
        self.assertEqual(summary["report"], str(good))
        self.assertAlmostEqual(summary["languages"]["ja"]["acceptance"], 9 / 13)
        self.assertEqual(summary["peak_GiB"], {"0": 1.0, "1": 1.5})
        line = out.strip()
        self.assertTrue(line.startswith("run-a "), line)
        self.assertIn("aggregate_decode_tps", line)
        self.assertLess(len(line.splitlines()), 4, "stdout must stay concise")

    def test_rejected_report_writes_no_artifact_and_exits_one(self):
        bad = self.write("run-b.json", report_of([(True, GROUP0, 4.0)], complete=False))
        code, _out, err = self.run_main([str(bad)])
        self.assertEqual(code, 1)
        self.assertIn("not complete", err)
        self.assertFalse((Path(self.td.name) / "run-b-summary.json").exists())

    def test_unreadable_or_non_json_input_is_reported_not_raised(self):
        code, _out, err = self.run_main([str(Path(self.td.name) / "nope.json")])
        self.assertEqual(code, 1)
        self.assertIn("cannot read", err)
        broken = self.write("broken.json", "this is not json")
        code, _out, err = self.run_main([str(broken)])
        self.assertEqual(code, 1)
        self.assertIn("not valid JSON", err)
        not_object = self.write("array.json", '[{"complete": true}]')
        code, _out, err = self.run_main([str(not_object)])
        self.assertEqual(code, 1)
        self.assertIn("must be a JSON object", err)

    def test_one_bad_report_does_not_stop_the_others(self):
        good = self.write("run-c.json", report_of([(True, GROUP1, 2.0)]))
        bad = self.write("run-d.json", report_of([(True, GROUP1, 2.0)], capacity_only=True))
        code, out, err = self.run_main([str(bad), str(good)])
        self.assertEqual(code, 1)
        self.assertIn("capacity_only", err)
        self.assertTrue((Path(self.td.name) / "run-c-summary.json").is_file())
        self.assertFalse((Path(self.td.name) / "run-d-summary.json").exists())
        self.assertIn("run-c", out)

    def test_help_exits_zero(self):
        code, out, _err = self.run_main(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("-m rocm_tools.rdna2.summarize_tp", out)

    def test_no_paths_is_a_usage_error(self):
        code, out, _err = self.run_main([])
        self.assertEqual(code, 2)
        self.assertIn("REPORT.json", out)


class Portability(unittest.TestCase):
    """Publication guards for the two helpers."""

    SOURCES = {name: Path(module.__file__).read_text()
               for name, module in (("power_server.py", power_server),
                                    ("summarize_tp.py", summarize_tp))}

    def test_no_personal_paths_or_sys_path_hacks(self):
        for name, src in self.SOURCES.items():
            for needle in ("/home/", "sys.path.insert", "datapool", "import torch"):
                with self.subTest(module=name, needle=needle):
                    self.assertNotIn(needle, src)

    def test_no_privilege_escalation_inside_the_helpers(self):
        for name, src in self.SOURCES.items():
            for needle in ("subprocess", "os.system", "sudo(", "Popen"):
                with self.subTest(module=name, needle=needle):
                    self.assertNotIn(needle, src)

    def test_power_server_touches_exactly_one_sysfs_attribute(self):
        src = self.SOURCES["power_server.py"]
        self.assertIn('POWER_LEVEL_ATTR = "power_dpm_force_performance_level"', src)
        self.assertIn('SYSFS_PCI_ROOT = "/sys/bus/pci/devices"', src)

    def test_summarizer_reuses_the_core_aggregator(self):
        src = self.SOURCES["summarize_tp.py"]
        self.assertIn("from rocm_tools.rdna2.tp_run import group_throughput_metrics", src)
        self.assertNotIn("def group_throughput_metrics", src)

    def test_entry_points_work_without_gpu_or_root(self):
        repo = str(Path(__file__).resolve().parents[3])
        env = {**os.environ, "PYTHONPATH": repo,
               "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
        cases = ([sys.executable, "-m", "rocm_tools.rdna2.power_server", "--help"], "--devices")
        cases2 = ([sys.executable, "-m", "rocm_tools.rdna2.summarize_tp", "--help"],
                  "summarize_tp REPORT.json")
        for argv, needle in (cases, cases2):
            with self.subTest(argv=argv[-1]):
                proc = subprocess.run(argv, cwd=repo, env=env, capture_output=True,
                                      text=True, timeout=180)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(needle, proc.stdout)


if __name__ == "__main__":
    unittest.main()
