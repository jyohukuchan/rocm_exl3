# SPDX-License-Identifier: MIT
# Copyright (c) 2026 jyohukuchan
"""Launch the downloaded JEV pack with a local rocm_exl3 checkout."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-dir', required=True)
    args, server_args = parser.parse_known_args()
    engine = Path(args.engine_dir).expanduser().resolve()
    if not (engine/'rocm_tools/jev_server.py').is_file():
        parser.error('Engine directory must contain rocm_tools/jev_server.py')
    sys.path.insert(0, str(engine))
    sys.argv = [str(Path(__file__)), '-m', str(Path(__file__).resolve().parent), *server_args]
    from rocm_tools.jev_server import main as serve
    serve()


if __name__ == '__main__':
    main()
