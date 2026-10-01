"""Native per-request timings for UI clients, with no GPU queries."""
import math


def inference_metrics(final):
    def count(name):
        value = final.get(name, 0)
        return max(0, int(value)) if isinstance(value, (int, float)) and math.isfinite(value) else 0

    def seconds(name):
        value = final.get(name)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None

    prompt = count("prompt_tokens")
    cached = min(prompt, count("cached_tokens"))
    output = count("new_tokens")
    prefill, decode = seconds("time_prefill"), seconds("time_generate")
    return {
        "version": 1,
        "prompt_tokens": prompt, "cached_tokens": cached,
        "prefill_tokens": prompt - cached, "output_tokens": output,
        "prefill_seconds": prefill, "decode_seconds": decode,
        "prefill_tokens_per_second": (prompt - cached) / prefill if prefill else None,
        "output_tokens_per_second": output / decode if decode else None,
    }
