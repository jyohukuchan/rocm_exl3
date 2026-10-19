"""Prediction fidelity on the reference model's trajectory."""

import math

import torch


class PredictionNMSE:
    """Compare complete target predictions for a cache-free image transformer.

    The observer returns None, leaving the teacher output and scheduler intact.
    Candidate and teacher receive the same latent and conditioning, including
    the flow-map target timestep. The transformer must return target tokens only.
    """

    def __init__(self, candidate_forward):
        self.candidate_forward = candidate_forward
        self.records = []

    @torch.inference_mode()
    def observe(self, module, inputs, kwargs, output):
        if inputs:
            raise ValueError("NMSE requires keyword transformer inputs")
        candidate = self.candidate_forward(**kwargs)
        reference = output[0] if isinstance(output, tuple) else output.sample
        candidate = candidate[0] if isinstance(candidate, tuple) else candidate.sample
        if candidate.shape != reference.shape or reference.shape != kwargs["hidden_states"].shape:
            raise ValueError("NMSE requires matching target-token prediction shapes")
        reference, candidate = reference.float(), candidate.float()
        error = (candidate - reference).square().sum(dtype=torch.float64).item()
        signal = reference.square().sum(dtype=torch.float64).item()
        if not math.isfinite(error) or not math.isfinite(signal) or signal <= 0:
            raise ValueError("NMSE requires finite predictions and positive reference energy")
        self.records.append({
            "step": len(self.records), "elements": reference.numel(),
            "timestep": kwargs["timestep"].float().cpu().tolist(),
            "target_timestep": kwargs["target_timestep"].float().cpu().tolist(),
            "squared_error_sum": error, "reference_squared_sum": signal, "nmse": error / signal,
        })

    def report(self):
        if not self.records:
            raise ValueError("No NMSE observations")
        error = math.fsum(r["squared_error_sum"] for r in self.records)
        signal = math.fsum(r["reference_squared_sum"] for r in self.records)
        return {"nmse": error / signal, "squared_error_sum": error,
                "reference_squared_sum": signal, "elements": sum(r["elements"] for r in self.records),
                "trajectory": "bf16", "prediction_scope": "target image tokens before scheduler",
                "aggregation": "sum_squared_error / sum_reference_squared; all target elements and steps",
                "steps": self.records}
