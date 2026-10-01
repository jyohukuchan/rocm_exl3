"""Exercise real process ownership, retries and health recovery without GPUs."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest


ENGINE = '''
import pathlib,subprocess,sys,http.server,json
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
p=root/'starts'; n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n))
flag=root/'unhealthy'; flag.unlink(missing_ok=True)
if n==1 and mode=='crash':
    worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    (root/'orphan').write_text(str(worker.pid))
    sys.exit(7)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(503 if flag.exists() else 200); self.end_headers()
        self.wfile.write(json.dumps({'status':'unavailable' if flag.exists() else 'ok'}).encode())
    def log_message(self,*args): pass
http.server.HTTPServer(('127.0.0.1',int(sys.argv[3])),Handler).serve_forever()
'''


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.03)
    raise AssertionError("Test process did not reach the expected state")


def report(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def alive(pid):
    try:
        # A zombie has exited and holds no GPU state, even when a container PID1
        # has not reaped it. Distinguish that from a surviving engine worker.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("mode", ["crash", "health"])
def test_supervisor_recovers_and_cleans_only_its_engine_group(tmp_path, mode):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    engine = tmp_path / 'engine.py'
    engine.write_text(ENGINE)
    status = tmp_path / 'status.json'
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    supervisor = subprocess.Popen([
        sys.executable, '-m', 'rocm_tools.exl3_server.supervisor',
        '--health-url', f'http://127.0.0.1:{port}/health', '--status-file', str(status),
        '--health-timeout', '.1', '--startup-timeout', '2', '--unhealthy-timeout', '.15',
        '--poll-interval', '.03', '--stop-grace', '.15', '--restart-delay', '.03',
        '--max-restarts', '2', '--', sys.executable, str(engine), str(tmp_path), mode, str(port)])
    try:
        wait_for(lambda: report(status).get('state') == 'ready')
        if mode == 'health':
            (tmp_path / 'unhealthy').touch()
        state = wait_for(lambda: (r if r.get('state') == 'ready' and r.get('restarts') == 1 else None)
                         if (r := report(status)) else None)
        assert (tmp_path / 'starts').read_text() == '2'
        if mode == 'crash':
            wait_for(lambda: not alive(int((tmp_path / 'orphan').read_text())))
        assert unrelated.poll() is None
        supervisor.terminate()
        assert supervisor.wait(timeout=5) == 0
        assert report(status)['state'] == 'stopped'
        assert not alive(state['pid'])
        assert unrelated.poll() is None
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=5)
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_supervisor_exhausts_its_restart_budget(tmp_path):
    status = tmp_path / 'status.json'
    result = subprocess.run([
        sys.executable, '-m', 'rocm_tools.exl3_server.supervisor', '--health-url', 'http://127.0.0.1:1',
        '--health-timeout', '.03', '--startup-timeout', '.1', '--unhealthy-timeout', '.1',
        '--poll-interval', '.03', '--stop-grace', '.1', '--restart-delay', '.03',
        '--max-restarts', '1', '--status-file', str(status), '--', sys.executable, '-c', 'raise SystemExit(7)'],
        timeout=5)
    assert result.returncode == 1
    assert report(status)['restarts'] == 1
    assert report(status)['state'] == 'failed'
