# SPDX-License-Identifier: MIT
"""Build a host-only GEMM grouping bridge without replacing the engine binary."""
import argparse
import hashlib
import json
import subprocess
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths,library_paths

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
source=Path(__file__).with_name('moe_group_sidecar.cpp')
output=args.output_dir/('exl3_moe_group_ext'+sysconfig.get_config_var('EXT_SUFFIX'))
command=['c++','-std=c++17','-shared','-fPIC','-O2','-DTORCH_EXTENSION_NAME=exl3_moe_group_ext','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),*[f'-I{p}' for p in include_paths()],'-I'+sysconfig.get_path('include'),*[f'-L{p}' for p in library_paths()],*[f'-Wl,-rpath,{p}' for p in library_paths()],str(source),'-lc10','-ltorch_cpu','-ltorch_python','-ldl','-o',str(output)]
subprocess.run(command,check=True)
(args.output_dir/'build.json').write_text(json.dumps({'command':command,'torch':torch.__version__,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'host_only':True},indent=2)+'\n')
print(output)
