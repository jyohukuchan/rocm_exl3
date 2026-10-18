import copy
import json
import pytest

from rocm_tools.qwen_image21 import source
from rocm_tools.qwen_image21_suite import matching_report


def test_resume_rejects_changed_sampling_and_invalid_measurements(tmp_path):
    sigmas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]
    (tmp_path / "model_index.json").write_text(json.dumps({"sample_sigmas": sigmas}))
    case = {"prompt": "A boat.", "seed": 17, "width": 1024, "height": 1024}
    report = {
        "source": source("turbo"), "precision": "bf16", "calibration_instrumented": False,
        "settings": {**case, "image": [], "negative_prompt": None, "true_cfg_scale": 1.0,
                     "use_kv_cache": True, "sample_sigmas": sigmas},
        "measurements": [{"warmup": False, "actual_steps": 8, "image_size": [1024, 1024],
                          "pipeline_seconds": 12.0, "denoising_ms": 8000.0,
                          "transformer_ms": [1000.0] * 8, "timesteps": [[s] for s in sigmas],
                          "cache_modes": ["extract"] + ["cached"] * 7}],
    }
    assert matching_report(report, case, [], "turbo", tmp_path, "bf16")
    wrong_seed = copy.deepcopy(report)
    wrong_seed["settings"]["seed"] += 1
    assert not matching_report(wrong_seed, case, [], "turbo", tmp_path, "bf16")
    wrong_schedule = copy.deepcopy(report)
    wrong_schedule["settings"]["sample_sigmas"] = None
    assert not matching_report(wrong_schedule, case, [], "turbo", tmp_path, "bf16")
    bad_timing = copy.deepcopy(report)
    bad_timing["measurements"][0]["denoising_ms"] = float("nan")
    assert not matching_report(bad_timing, case, [], "turbo", tmp_path, "bf16")
    no_cache = copy.deepcopy(report)
    no_cache["measurements"][0]["cache_modes"] = [None] * 8
    assert not matching_report(no_cache, case, [], "turbo", tmp_path, "bf16")


def test_compare_rejects_different_offload_or_prompt_cache():
    from rocm_tools.qwen_image21_compare import paired_reports

    left = {"settings": {"seed": 17, "offload": "none", "reconstruction_workspace_bytes": None},
            "source": {"revision": "same"}, "versions": {"torch": "same"},
            "gpu": "R9700", "arch": "gfx1201", "text_encoding": {"sha": "same"},
            "implementation": {"runtime": "same"}}
    right = copy.deepcopy(left)
    right["settings"]["reconstruction_workspace_bytes"] = 256 * 1024**2
    paired_reports(left, right)
    right["settings"]["offload"] = "model"
    with pytest.raises(ValueError, match="settings"):
        paired_reports(left, right)
    right["settings"]["offload"] = "none"
    right["text_encoding"]["sha"] = "other"
    with pytest.raises(ValueError, match="encodings"):
        paired_reports(left, right)
