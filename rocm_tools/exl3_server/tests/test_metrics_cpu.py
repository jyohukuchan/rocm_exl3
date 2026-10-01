from rocm_tools.exl3_server.metrics import inference_metrics, strip_timing_footer, timing_footer


def test_rates_use_uncached_tokens_and_native_phase_times():
    result = inference_metrics(dict(prompt_tokens=1000, cached_tokens=800,
                                    new_tokens=50, time_prefill=2, time_generate=5))
    assert result["prefill_tokens"] == 200
    assert result["prefill_tokens_per_second"] == 100
    assert result["output_tokens_per_second"] == 10


def test_missing_or_invalid_time_does_not_make_a_rate():
    for value in [None, 0, -1, float("nan"), float("inf"), "unknown"]:
        result = inference_metrics(dict(time_prefill=value, time_generate=value))
        assert result["prefill_tokens_per_second"] is None
        assert result["output_tokens_per_second"] is None


def test_complete_cache_hit_and_invalid_counts():
    result = inference_metrics(dict(prompt_tokens=100, cached_tokens=999, time_prefill=.1))
    assert result["cached_tokens"] == 100
    assert result["prefill_tokens_per_second"] == 0
    assert inference_metrics(dict(prompt_tokens=float("inf")))["prompt_tokens"] == 0


def test_footer_missing_rates_and_exact_suffix_only():
    footer = timing_footer(inference_metrics({}, total_seconds=2.5))
    assert "Prefill: N/A tok/s | Decode: N/A tok/s | Total: 2.50 s" in footer
    assert strip_timing_footer("answer\n" + footer + "\n") == "answer\n"
    assert strip_timing_footer("answer" + footer + footer) == "answer"
    assert strip_timing_footer("quote" + footer + "\nmore") == "quote" + footer + "\nmore"
    assert strip_timing_footer("answer\nPrefill: 100 tok/s") == "answer\nPrefill: 100 tok/s"
