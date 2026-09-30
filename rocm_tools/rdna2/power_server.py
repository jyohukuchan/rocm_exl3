#!/usr/bin/env python3
"""Repo-native privileged two-GPU DPM power SERVER (stdlib only, no GPU work).

Promotes the private one-off `power_switch_server.py` used for the context-batch
runs into a reproducible, published helper that speaks EXACTLY the protocol
`rocm_tools/rdna2/power_policy.py` already implements:

    request : one JSON line  {"mode": "auto"|"profile_peak", "label": <any>}
    reply   : one JSON line event with label / mode / before / after /
              requested_unix_s / per_gpu_write_ms / both_write_ms /
              write_and_readback_ms

Nothing else may be written, read or switched: the only sysfs attribute this
helper touches is `<SYSFS_PCI_ROOT>/<BDF>/power_dpm_force_performance_level`,
derived from validated BDFs. There is deliberately no `--sysfs-root`, no
`--attr`, no device-list file, no arbitrary path on the command line.

Run it EXPLICITLY under sudo -- the helper never spawns sudo itself, and it does
no GPU work, so it stays usable on a machine with no ROCm installed:

    sudo python3 rocm_tools/rdna2/power_server.py \
        --devices 0000:43:00.0 0000:03:00.0 \
        --socket  /srv/rdna2-bench/power.sock \
        --report  /srv/rdna2-bench/power.json

(`--socket`/`--report` parent directories are created 0700 and handed to the
invoking user if missing; pick a location the inference process can read.)

Security / robustness contract:
  * exactly two distinct canonical lowercase PCI BDFs (DDDD:BB:DD.F);
  * the socket path must NOT already exist -- a pre-existing socket is refused,
    never unlinked, because deleting someone else's socket is how you hijack a
    peer; the bound socket is chmod 0600;
  * when launched through sudo, the socket (and the report, and any directory
    this helper had to create) is chowned to SUDO_UID/SUDO_GID so the
    UNPRIVILEGED inference process can connect and later read its own artifact;
  * a pre-existing report file is refused and a symlinked report/socket is
    refused; updates land through an atomic rename of a fresh 0600 temp file;
  * ONE client connection, then the helper exits on its disconnect;
  * the original modes are captured BEFORE anything is switched, and restored
    on client disconnect, malformed/invalid request, failed sysfs write,
    readback mismatch, SIGINT, SIGTERM or any other error. Further INT/TERM are
    ignored only while restoring, so a second signal cannot leave a GPU pinned.

`serve()` and friends take paths directly (internal API, used by the CPU tests
with a fake sysfs tree); only `main()` maps CLI BDFs onto the real sysfs root.

Exit codes: 0 clean disconnect and confirmed restore, 1 modes NOT restored
(check dmesg / re-run by hand -- treat the run as uncontrolled), 2 session error
(bad request, invalid mode, sysfs write or readback failure), 3 setup refused
(bad BDF, missing attribute, pre-existing socket/report, symlink, bind failure).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import stat
import sys
import time
from pathlib import Path

MODES = ("auto", "profile_peak")
POWER_LEVEL_ATTR = "power_dpm_force_performance_level"
SYSFS_PCI_ROOT = "/sys/bus/pci/devices"
NUM_DEVICES = 2
SOCKET_MODE = 0o600
FILE_MODE = 0o600                     # artifacts are root-created, then chowned
DIR_MODE = 0o700

EXIT_CLEAN, EXIT_NOT_RESTORED, EXIT_SESSION, EXIT_SETUP = 0, 1, 2, 3

# Canonical Linux BDF as udev/sysfs name it: 4-hex domain, 2-hex bus, 2-hex
# device, 1-hex function, lowercase, dot before the function.
_BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]$")


class PowerServerError(Exception):
    """Validation, protocol, sysfs or filesystem failure. Fail closed."""


class _ProtocolError(PowerServerError):
    """A request could not be served. Carries the partial event, if any."""

    def __init__(self, message, event=None):
        super().__init__(message)
        self.event = event


class _SetupError(PowerServerError):
    """The endpoint could not be made ready; nothing has been switched yet."""


# Set while the restore is in flight so a second signal cannot interrupt it
# (muting signals globally would surprise in-process callers and tests).
_RESTORING = {"active": False}


class _Interrupted(Exception):
    def __init__(self, signum):
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


# ---------------------------------------------------------------------------
# validation / paths (pure, CPU-testable)
# ---------------------------------------------------------------------------

def validate_bdfs(values):
    """Exactly NUM_DEVICES distinct, canonical, lowercase PCI BDFs."""
    bdfs = list(values)
    if len(bdfs) != NUM_DEVICES:
        raise PowerServerError(f"exactly {NUM_DEVICES} PCI BDFs are required, got {len(bdfs)}: {bdfs}")
    for b in bdfs:
        if not _BDF_RE.match(b or ""):
            raise PowerServerError(f"'{b}' is not a canonical lowercase PCI BDF "
                                   "(expected DDDD:BB:DD.F such as 0000:43:00.0)")
    if len(set(bdfs)) != NUM_DEVICES:
        raise PowerServerError(f"the {NUM_DEVICES} PCI BDFs must be distinct: {bdfs}")
    return bdfs


def power_level_paths(bdfs, root=None):
    """The ONE attribute per validated BDF. `root` is internal (tests only)."""
    base = Path(SYSFS_PCI_ROOT if root is None else root)
    paths = [base / b / POWER_LEVEL_ATTR for b in bdfs]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise PowerServerError("power-level attribute not present: " + ", ".join(missing))
    return paths


def invoker_ids(env=None):
    """(uid, gid) to hand artifacts to when launched via sudo, else None."""
    env = os.environ if env is None else env
    raw_uid, raw_gid = env.get("SUDO_UID"), env.get("SUDO_GID")
    if raw_uid is None or raw_gid is None:
        return None
    try:
        uid, gid = int(raw_uid), int(raw_gid)
    except ValueError:
        raise PowerServerError(f"invalid SUDO_UID/SUDO_GID ({raw_uid!r}, {raw_gid!r})") from None
    if uid < 0 or gid < 0:
        raise PowerServerError(f"negative SUDO_UID/SUDO_GID ({uid}, {gid})")
    return uid, gid


def ensure_private_dir(directory, owner):
    """Create a missing output directory as 0700 and hand it to `owner`.

    Existing directories must be private and owned by the invoking user.
    Their permissions are checked, never changed."""
    d = Path(directory)
    if d.is_symlink():
        raise PowerServerError(f"refusing to use symlinked directory {d}")
    if d.is_dir():
        info = d.stat()
        expected_uid = owner[0] if owner else os.getuid()
        if info.st_uid != expected_uid or info.st_mode & 0o077:
            raise PowerServerError(f"{d} must be owned by uid {expected_uid} and private (0700); use mktemp -d")
        return
    missing = []
    cur = d
    while not cur.is_dir():
        if cur.is_symlink():
            raise PowerServerError(f"refusing to create under symlinked directory {cur}")
        missing.append(cur)
        if cur == cur.parent:
            break
        cur = cur.parent
    for new in reversed(missing):
        try:
            new.mkdir(mode=DIR_MODE)
            os.chmod(new, DIR_MODE)          # mkdir is umask-masked; be explicit
            if owner:
                os.chown(new, *owner)        # else the unprivileged client cannot traverse
        except OSError as e:
            raise PowerServerError(f"cannot create private directory {new}: {e!r}") from e


def _read_mode(path):
    return Path(path).read_text().strip()


def _write_mode(path, mode):
    Path(path).write_text(mode + "\n")


class _ReportFile:
    """Create-once, atomically updated 0600 JSON artifact."""

    def __init__(self, path, owner=None):
        self.path = Path(path)
        self.owner = owner
        self.created = False

    def _guard(self):
        if self.path.is_symlink():
            raise PowerServerError(f"refusing to write report through symlink {self.path}")

    def create(self, data):
        self._guard()
        if self.path.exists():
            raise PowerServerError(f"refusing to overwrite pre-existing report {self.path}; "
                                   "choose a fresh path (one artifact per session)")
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
            try:
                os.write(fd, (json.dumps(data, indent=2) + "\n").encode())
            finally:
                os.close(fd)
            if self.owner:
                os.chown(self.path, *self.owner)
        except OSError as e:
            raise PowerServerError(f"cannot create report {self.path}: {e!r}") from e
        self.created = True

    def save(self, data):
        if not self.created:
            return
        self._guard()
        tmp = Path(f"{self.path}.tmp{os.getpid()}")
        created = False
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
            created = True
            try:
                os.write(fd, (json.dumps(data, indent=2) + "\n").encode())
            finally:
                os.close(fd)
            if self.owner:
                os.chown(tmp, *self.owner)
            os.replace(tmp, self.path)       # replaces the link, never follows it
        except OSError as e:
            if created:
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise PowerServerError(f"cannot update report {self.path}: {e!r}") from e


# ---------------------------------------------------------------------------
# the switch itself
# ---------------------------------------------------------------------------

def apply_mode(paths, mode, label):
    """Switch every device to `mode` and verify by readback. Returns the event."""
    if mode not in MODES:
        raise _ProtocolError(f"invalid mode {mode!r}, expected one of {list(MODES)}")
    paths = [Path(p) for p in paths]
    try:
        before = [_read_mode(p) for p in paths]
    except OSError as e:
        raise _ProtocolError(f"cannot read the current mode: {e!r}") from e
    event = {"label": label, "mode": mode, "before": before, "after": None,
             "requested_unix_s": time.time(), "per_gpu_write_ms": [],
             "both_write_ms": None, "write_and_readback_ms": None}
    t0 = time.perf_counter_ns()
    for p in paths:
        a = time.perf_counter_ns()
        try:
            _write_mode(p, mode)
        except OSError as e:
            raise _ProtocolError(f"sysfs write to {p} failed: {e!r}", event) from e
        event["per_gpu_write_ms"].append((time.perf_counter_ns() - a) / 1e6)
    event["both_write_ms"] = (time.perf_counter_ns() - t0) / 1e6
    try:
        event["after"] = [_read_mode(p) for p in paths]
    except OSError as e:
        raise _ProtocolError(f"readback failed: {e!r}", event) from e
    event["write_and_readback_ms"] = (time.perf_counter_ns() - t0) / 1e6
    if any(v != mode for v in event["after"]):
        raise _ProtocolError(f"readback {event['after']} does not match requested "
                             f"mode {mode!r}; policy state is not trustworthy", event)
    return event


def restore_modes(paths, originals):
    """Write the captured originals back; returns (final_modes, errors)."""
    errors = []
    for p, value in zip(paths, originals):
        try:
            _write_mode(p, value)
        except Exception as e:
            errors.append(f"restore {p}: {e!r}")
    final = []
    for p in paths:
        try:
            final.append(_read_mode(p))
        except Exception as e:
            errors.append(f"readback {p}: {e!r}")
            final.append(None)
    return final, errors


# ---------------------------------------------------------------------------
# one-client Unix-socket session
# ---------------------------------------------------------------------------

def handle_client(conn, paths, report, rep):
    """Serve requests until the single client disconnects. Returns an exit code."""
    with conn.makefile("rwb", buffering=0) as fh:
        while True:
            try:
                line = fh.readline()
            except OSError as e:
                raise _ProtocolError(f"client link failed: {e!r}") from e
            if not line:
                return EXIT_CLEAN                 # client gone: normal end of session
            try:
                req = json.loads(line)
            except ValueError as e:
                raise _ProtocolError(f"request is not one JSON object per line ({e})") from e
            if not isinstance(req, dict):
                raise _ProtocolError("request must be a JSON object")
            try:
                event = apply_mode(paths, req.get("mode"), req.get("label"))
            except _ProtocolError as e:
                # Fail loud, but tell the client WHY: a reply without a confirmed
                # "after" makes power_policy.py refuse to label the run controlled.
                _send(fh, {**(e.event or {"label": req.get("label"), "mode": req.get("mode")}),
                           "error": str(e)})
                raise
            report["events"].append(event)
            rep.save(report)                      # durable before the client resumes work
            _send(fh, event)


def _send(fh, payload):
    try:
        fh.write((json.dumps(payload) + "\n").encode())
    except OSError:
        pass                                      # best effort; the client fails closed


def serve(paths, socket_path, report_path, owner=None):
    """Full privileged session: capture, bind, serve one client, ALWAYS restore.

    `paths` are power-level attribute paths (see `power_level_paths`). Raises
    PowerServerError for setup refusals, where nothing has been switched yet."""
    paths = [Path(p) for p in paths]
    socket_path, report_path = Path(socket_path), Path(report_path)
    originals = [_read_mode(p) for p in paths]           # BEFORE any write
    if socket_path.is_symlink():
        raise PowerServerError(f"refusing to use symlinked socket {socket_path}")
    if socket_path.exists():
        raise PowerServerError(f"socket {socket_path} already exists; refusing to delete "
                               "someone else's endpoint (remove a stale one yourself)")
    rep = _ReportFile(report_path, owner)
    report = {"pid": os.getpid(),
              "devices": [str(p) for p in paths],
              "socket": str(socket_path),
              "original": originals,
              "events": [],
              "restored": False,
              "started_unix_s": time.time()}
    rep.create(report)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn = None
    bound_identity = None
    code = EXIT_CLEAN
    try:
        try:
            srv.bind(str(socket_path))
            info = socket_path.lstat()
            bound_identity = (info.st_dev, info.st_ino)
        except OSError as e:
            raise _SetupError(f"cannot bind {socket_path}: {e!r}") from e
        try:
            if owner:
                os.chown(socket_path, *owner)
            os.chmod(socket_path, SOCKET_MODE)
        except OSError as e:
            raise _SetupError(f"cannot make {socket_path} private: {e!r}") from e
        srv.listen(1)                               # exactly one client, ever
        rep.save(report)
        print("READY", flush=True)
        try:
            conn, _ = srv.accept()
        except OSError as e:
            raise _ProtocolError(f"accept failed: {e!r}") from e
        code = handle_client(conn, paths, report, rep)
    except _Interrupted as e:
        code = 128 + e.signum
        report["interrupted"] = signal.Signals(e.signum).name
        report["error"] = f"interrupted by {signal.Signals(e.signum).name}"
    except _ProtocolError as e:
        code = EXIT_SESSION
        report["error"] = str(e)
        if e.event is not None:
            report["events"].append(e.event)
            rep.save(report)
    except _SetupError as e:
        code = EXIT_SETUP                             # nothing was switched yet
        report["error"] = str(e)
        raise
    except Exception as e:
        code = EXIT_SESSION
        report["error"] = repr(e)
    finally:
        _RESTORING["active"] = True                 # a second signal must not pin a GPU
        try:
            final, errors = restore_modes(paths, originals)
        finally:
            _RESTORING["active"] = False
        report["final"] = final
        report["restore_errors"] = errors
        report["restored"] = final == originals and not errors
        report["finished_unix_s"] = time.time()
        try:
            if conn:
                conn.close()
        finally:
            srv.close()
        cleanup = []
        try:
            if bound_identity is not None:
                try:
                    info = socket_path.lstat()
                except FileNotFoundError:
                    info = None
                # Unlink ONLY the inode we bound: if someone planted another file at
                # this path mid-session, deleting the path would delete THEIRS.
                same_inode = info is not None and (info.st_dev, info.st_ino) == bound_identity
                if same_inode and stat.S_ISSOCK(info.st_mode):
                    socket_path.unlink()
        except OSError as e:
            cleanup.append(f"socket cleanup: {e!r}")
        report["cleanup_errors"] = cleanup
        try:
            rep.save(report)
        except PowerServerError as e:
            print(f"power_server: {e}", file=sys.stderr)
        print(f"RESTORED {report['restored']}", flush=True)
    if report.get("error") or report["restore_errors"] or report["cleanup_errors"]:
        print("power_server: " + json.dumps({"error": report.get("error"),
                                             "restore_errors": report["restore_errors"],
                                             "cleanup_errors": report["cleanup_errors"]}),
              file=sys.stderr)
    return code, report


def interrupt_handler(signum, frame):
    """Signal handler: raise so `serve()`'s finally restores. Stays quiet while the
    restore is already running, so a second Ctrl-C cannot leave a GPU pinned."""
    if not _RESTORING["active"]:
        raise _Interrupted(signum)


def install_signal_handlers():
    """CLI-only (signals must be installed from the main thread). Returns previous."""
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, interrupt_handler)
    return previous


def restore_signal_handlers(previous):
    for sig, handler in (previous or {}).items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError, TypeError):
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        prog="power_server.py",
        description="Privileged two-GPU PCI DPM power server for the RDNA2 TP2 "
                    "harness: one Unix-socket client, JSON-line auto/profile_peak "
                    "switches with readback confirmation, originals restored on exit.",
        epilog="Run this explicitly under sudo; it never spawns sudo and never touches "
               "anything but <SYSFS_PCI_ROOT>/<BDF>/" + POWER_LEVEL_ATTR + ". The socket is "
               "chmod 0600 and, when sudoed, chowned to SUDO_UID/SUDO_GID so the "
               "unprivileged inference process can connect. Pre-existing socket or report "
               "paths, and symlinks, are refused rather than replaced. Exit codes: "
               f"{EXIT_CLEAN} clean, {EXIT_NOT_RESTORED} modes not restored, "
               f"{EXIT_SESSION} session error, {EXIT_SETUP} setup refused.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs=NUM_DEVICES, metavar="PCI_BDF", required=True,
                    help=f"exactly {NUM_DEVICES} distinct canonical lowercase PCI BDFs "
                         "(DDDD:BB:DD.F), e.g. 0000:43:00.0 0000:03:00.0")
    ap.add_argument("--socket", required=True,
                    help="private Unix socket path (must not exist yet; created 0600)")
    ap.add_argument("--report", required=True,
                    help="JSON session report path (must not exist yet; created 0600)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    previous = None
    try:
        paths = power_level_paths(validate_bdfs(args.devices))
        socket_path = Path(args.socket).expanduser()
        report_path = Path(args.report).expanduser()
        owner = invoker_ids()
        for directory in (socket_path.parent, report_path.parent):
            ensure_private_dir(directory, owner)
        previous = install_signal_handlers()
        code, report = serve(paths, socket_path, report_path, owner=owner)
    except PowerServerError as e:
        print(f"power_server: error: {e}", file=sys.stderr)
        return EXIT_SETUP
    except OSError as e:
        print(f"power_server: error: {e!r}", file=sys.stderr)
        return EXIT_SETUP
    finally:
        restore_signal_handlers(previous)
    if not report["restored"]:
        print("power_server: GPU power modes were NOT restored -- treat any run in this "
              "window as uncontrolled", file=sys.stderr)
        return EXIT_NOT_RESTORED
    return code


if __name__ == "__main__":
    sys.exit(main())
