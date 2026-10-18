"""Teacher-trajectory DiT output NMSE, with independent candidate prefix caches."""

import math

import torch


class OutputNMSE:
    """Observe BF16 calls without feeding candidate predictions to the scheduler.

    Only target-image tokens are scored, including on the prefix-extraction call.
    Each model computes its own prefix KV cache from identical conditioning.
    ``candidate_forward`` can be the reference's bound forward for a self check;
    calling it directly bypasses the reference's outer observation hook.
    """

    def __init__(self, candidate_forward, cache_factory):
        self.candidate_forward = candidate_forward
        self.cache_factory = cache_factory
        self.caches = {}
        self.step = 0
        self.records = []

    @torch.inference_mode()
    def observe(self, module, inputs, kwargs, output):
        if inputs:
            raise ValueError("NMSE requires keyword transformer inputs")
        candidate_kwargs = dict(kwargs)
        reference_cache = kwargs.get("kv_cache")
        mode = kwargs.get("kv_cache_mode")
        if reference_cache is not None:
            key = id(reference_cache)
            if mode == "extract":
                # Keep the source object alive: ids must not be reused between branches.
                self.caches[key] = (reference_cache, self.cache_factory())
            if key not in self.caches:
                raise ValueError("NMSE cached call without prefix extraction")
            candidate_kwargs["kv_cache"] = self.caches[key][1]
        candidate = self.candidate_forward(**candidate_kwargs)
        reference = output[0] if isinstance(output, tuple) else output.sample
        candidate = candidate[0] if isinstance(candidate, tuple) else candidate.sample
        if reference.shape != candidate.shape:
            raise ValueError("NMSE candidate/reference shape mismatch")
        target_tokens = math.prod(kwargs["img_shapes"][0][-1])
        reference = reference[:, -target_tokens:].float()
        candidate = candidate[:, -target_tokens:].float()
        if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
            raise ValueError("Nonfinite NMSE prediction")
        # FP32 subtraction/squaring, FP64 reductions and Python double totals.
        error = (candidate - reference).square().sum(dtype=torch.float64).item()
        signal = reference.square().sum(dtype=torch.float64).item()
        if not math.isfinite(error) or not math.isfinite(signal) or signal <= 0:
            raise ValueError("NMSE requires finite errors and positive reference energy")
        self.records.append({"step": self.step, "timestep": kwargs["timestep"].float().cpu().tolist(),
                             "cache_mode": mode, "elements": reference.numel(),
                             "squared_error_sum": error, "reference_squared_sum": signal,
                             "nmse": error / signal})
        # Returning None leaves the original BF16 pipeline output untouched.

    def report(self):
        if not self.records:
            raise ValueError("No NMSE observations")
        error = math.fsum(r["squared_error_sum"] for r in self.records)
        signal = math.fsum(r["reference_squared_sum"] for r in self.records)
        return {"nmse": error / signal, "squared_error_sum": error,
                "reference_squared_sum": signal, "elements": sum(r["elements"] for r in self.records),
                "aggregation": "sum_squared_error / sum_reference_squared; all target elements and steps",
                "trajectory": "bf16", "prediction_scope": "target image tokens, before scheduler/CFG",
                "prefix_cache_policy": "independent per model, identical conditioning",
                "steps": self.records}
