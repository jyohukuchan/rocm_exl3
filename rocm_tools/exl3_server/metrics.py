"""Native per-request timings for UI clients, with no GPU queries."""
import math
import re


# Recognize only our exact, terminal footer. Assistant history is cleaned at the
# template boundary even if display is now disabled; ordinary timing prose stays.
_VALUE = r"(?:[0-9]+\.[0-9]{2}|N/A)"
_RATES = (r"\n\*Prefill: " + _VALUE + r" tok/s \| Decode: " + _VALUE
          + r" tok/s \| Total: " + _VALUE + r" s")
_END = r"\*\n<!-- /exl3-timings -->[ \t\r\n]*\Z"
_FOOTERS = [
    re.compile(r"\n\n<!-- exl3-timings:v1 -->" + _RATES + _END),
    re.compile(r"\n\n<!-- exl3-timings:v2 -->" + _RATES
               + r" \| Draft: (?:[0-9]+\.[0-9]{2}%|N/A)" + _END),
]


def strip_timing_footer(text):
    while True:
        match = next((m for pattern in _FOOTERS if (m := pattern.search(text))), None)
        if match is None:
            return text
        text = text[:match.start()]


def timing_footer(metrics):
    def display(name):
        value = metrics.get(name)
        return f"{value:.2f}" if isinstance(value, (int, float)) and math.isfinite(value) else "N/A"

    acceptance = metrics.get("draft_acceptance_rate")
    draft = f"{acceptance * 100:.2f}%" if isinstance(acceptance, (int, float)) \
        and math.isfinite(acceptance) else "N/A"
    return ("\n\n<!-- exl3-timings:v2 -->\n"
            f"*Prefill: {display('prefill_tokens_per_second')} tok/s | "
            f"Decode: {display('output_tokens_per_second')} tok/s | "
            f"Total: {display('total_seconds')} s | Draft: {draft}*\n<!-- /exl3-timings -->")


def inference_metrics(final, *, total_seconds=None):
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
    accepted, rejected = count("accepted_draft_tokens"), count("rejected_draft_tokens")
    proposed = accepted + rejected
    result = {
        "version": 1,
        "prompt_tokens": prompt, "cached_tokens": cached,
        "prefill_tokens": prompt - cached, "output_tokens": output,
        "prefill_seconds": prefill, "decode_seconds": decode,
        "prefill_tokens_per_second": (prompt - cached) / prefill if prefill else None,
        "output_tokens_per_second": output / decode if decode else None,
        "accepted_draft_tokens": accepted, "rejected_draft_tokens": rejected,
        "draft_tokens": proposed,
        "draft_acceptance_rate": accepted / proposed if proposed else None,
    }
    if total_seconds is not None:
        result["total_seconds"] = max(0., float(total_seconds))
    return result
