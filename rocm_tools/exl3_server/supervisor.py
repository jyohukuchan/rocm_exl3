"""Own one engine process group and recover boundedly after exit or failed health.

Run this in the same PID namespace as the engine (inside its container). It must
not wrap `docker exec`, whose engine descendants belong to another namespace.
No GPU reset, system-wide process search, or unrelated process cleanup is used.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def healthy(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
            return response.status == 200 and isinstance(payload, dict) and payload.get("status") == "ok"
    except (OSError, ValueError):
        return False


def stop_group(process, grace):
    # Popen(start_new_session=True) created this group exclusively for this
    # engine. Kill its workers even if the parent exited before cleanup.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader, including on an intentional shutdown.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(args):
    stopping = False

    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    report = {"restarts": 0, "state": "starting"}

    def save(state, **fields):
        report.update(state=state, **fields)
        if args.status_file:
            target = Path(args.status_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            with temporary.open("w") as output:
                os.chmod(temporary, 0o600)
                json.dump(report, output)
            temporary.replace(target)

    try:
        while not stopping:
            child = subprocess.Popen(args.command, start_new_session=True)
            started = time.monotonic()
            last_healthy = None
            reason = "engine_exited"
            try:
                save("starting", pid=child.pid)
                while not stopping and child.poll() is None:
                    now = time.monotonic()
                    if healthy(args.health_url, args.health_timeout):
                        last_healthy = time.monotonic()
                        save("ready")
                    elif last_healthy is None and now - started >= args.startup_timeout:
                        reason = "startup_timeout"
                        break
                    elif last_healthy is not None and now - last_healthy >= args.unhealthy_timeout:
                        reason = "health_timeout"
                        break
                    else:
                        save("starting" if last_healthy is None else "unhealthy")
                    # Responsive shutdown even with a long configured poll interval.
                    until = time.monotonic() + args.poll_interval
                    while not stopping and time.monotonic() < until:
                        time.sleep(min(.1, max(0., until - time.monotonic())))
            finally:
                exit_code = child.poll()
                stop_group(child, args.stop_grace)
            if stopping:
                save("stopped")
                return 0
            save("failed", reason=reason, exit_code=exit_code)
            if report["restarts"] >= args.max_restarts:
                return 1
            save("restarting", restarts=report["restarts"] + 1)
            until = time.monotonic() + args.restart_delay
            while not stopping and time.monotonic() < until:
                time.sleep(min(.1, max(0., until - time.monotonic())))
        save("stopped")
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--health-timeout", type=float, default=5)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--unhealthy-timeout", type=float, default=90)
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--stop-grace", type=float, default=15)
    parser.add_argument("--restart-delay", type=float, default=10)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--status-file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command or args.max_restarts < 0 or any(
        not 0 < getattr(args, name) < float("inf") for name in
        ("health_timeout", "startup_timeout", "unhealthy_timeout", "poll_interval", "stop_grace", "restart_delay")):
        parser.error("A command and finite positive timeouts/delays are required")
    raise SystemExit(supervise(args))


if __name__ == "__main__":
    main()
