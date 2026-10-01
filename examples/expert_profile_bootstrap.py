"""Opt-in Qwen3.8 TP route capture; run before the server's __main__ module.

Requires EXL3_EXPERT_PROFILE_DIR and EXL3_TP_REPLICATE_ROUTER=1.
Never use this bootstrap for performance measurements.
"""
import json
import os
from pathlib import Path

from rocm_tools.exl3_server import runtime
from rocm_tools.rdna2 import expert_route_profile

directory = Path(os.environ["EXL3_EXPERT_PROFILE_DIR"])
original_load = runtime.load_runtime


def load_with_profile(*args, **kwargs):
    result = original_load(*args, **kwargs)
    try:
        if not result.model.loaded_tp:
            raise ValueError("Expert route profiling requires a TP model")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "plan.json").write_text(json.dumps(result.model.plan, indent=2) + "\n")
        result.model.tp_worker_dispatch_wait_multi(result.model.active_devices,
                                                  expert_route_profile.install, (str(directory), 4096))
    except BaseException:
        result.unload_models()
        raise
    unload = result.unload_models

    def close():
        try:
            if result.model.loaded_tp:
                result.model.tp_worker_dispatch_wait_multi(result.model.active_devices,
                                                          expert_route_profile.flush, ())
        finally:
            unload()

    result.unload_models = close
    runtime.load_runtime = original_load
    return result


runtime.load_runtime = load_with_profile
