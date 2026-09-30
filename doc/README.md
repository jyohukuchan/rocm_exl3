# Documentation

Start with [the V620 build and reproduction guide](reproduce_v620.md), [changes from CarouselAether/rocm_exl3](fork_changes.md), and the [2026-09-30 benchmark evidence](../benchmarks/2026-09-30/README.md).

## Validated behavior and investigations

- [V620 TP2 context, batch and MTP window measurements](qwen38_v620_context_batch.md)
- [TP decode optimization and K5/V4 selection](v620_tp_decode_optimization.md)
- [MTP results](qwen38_v620_mtp.md) and [official-source MTP precision comparison](qwen38_v620_mtp_precision.md)
- [RDNA multirow padded-load bounds fix](rdna2_multirow_bounds_fix.md)
- [R9700 versus V620 measurements and workarounds](r9700_vs_v620.md)
- [Single-V620 validation](rdna2_phase2_results.md) and [two-V620 layer split](v620_pair_results.md)
- [Power-policy switching](qwen38_v620_phase_switch.md) and [power comparison](qwen38_v620_power.md)

These investigation reports are historical records. Local artifact tags, source snapshots and example container paths in them are not all distributed. Current runnable entry points are in the reproduction guide; the dated public bundle identifies exactly which inputs and raw reports are included. Do not treat an old run's environment-specific JSON configuration as automatically loaded software defaults.

The fork remains experimental: documented workloads and hardware are verified, while general CUDA/RDNA-family compatibility, every model/server path, and maximum batch2–4 context windows are not established by these measurements.
