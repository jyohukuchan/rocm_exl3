"""Synthetic full-cache memory probe; no claim of long-context model quality.

Call exercise(runtime, report_path) after the normal server runtime has loaded.
The caller must discard this runtime afterwards: synthetic history replaces KV.
Worker helpers remain importable by name for the spawned TP ranks.
"""
from pathlib import Path
import json
import threading
import time


def initialize_rank_cache(local_context, cache_id):
    import torch
    from rocm_tools.rdna2.multi_gpu import _iter_module_tree

    device = local_context["device"]
    layers, seen = [], set()
    with torch.cuda.device(device):
        for module in local_context.get("modules") or []:
            for child in _iter_module_tree(module):
                for layer in getattr(child, "cache_layers", []):
                    if layer.cache_id != cache_id or id(layer) in seen:
                        continue
                    seen.add(id(layer))
                    for tensor in layer.get_tensors():
                        tensor.zero_()
                    layer.sk.fill_(.001)
                    layer.sv.fill_(.001)
                    # Distinct finite pooled scores avoid a degenerate all-zero
                    # history while needing only a small temporary range tensor.
                    pooled = layer.pooled.view(-1, layer.index_head_dim)
                    for start in range(0, pooled.shape[0], 4096):
                        end = min(start + 4096, pooled.shape[0])
                        pooled[start:end, 0].copy_(
                            torch.arange(start, end, device=device).float().mul_(1e-6).half())
                    layers.append({"pages": int(layer.qk.shape[0]),
                                   "tokens": int(layer.shape[0] * layer.shape[1]),
                                   "bytes": int(layer.storage_size())})
        torch.cuda.synchronize(device)
    return {"device": device, "layers": layers}


def exercise(runtime, report_path, pci_devices=("0000:43:00.0", "0000:03:00.0")):
    import torch
    from exllamav3.generator import Generator, Job
    from exllamav3.generator.sampler import ArgmaxSampler
    from rocm_tools.rdna2 import tp_run

    capacity = runtime.cache.max_num_tokens
    output_tokens = 32
    draft_tokens = int(runtime.generator_kwargs.get("num_draft_tokens") or 4)
    prompt_tokens = capacity - output_tokens - 1 - draft_tokens
    report_path = Path(report_path)
    report = {"capacity_tokens": capacity, "prompt_tokens": prompt_tokens,
              "generated_tokens_requested": output_tokens, "draft_tokens": draft_tokens,
              "synthetic_history": True, "quality_evaluation": False, "stages": []}
    done = threading.Event()
    peaks = [0] * len(pci_devices)
    samples = []
    boards = [Path("/sys/bus/pci/devices") / pci / "mem_info_vram_used"
              for pci in pci_devices]
    assert len(boards) == len(runtime.model.active_devices)
    start = time.monotonic()

    def monitor():
        while not done.is_set():
            values = [int(p.read_text()) for p in boards]
            for i, value in enumerate(values):
                peaks[i] = max(peaks[i], value)
            samples.append({"seconds": time.monotonic() - start, "vram_bytes": values})
            done.wait(.05)

    def save():
        report["board_peak_bytes"] = peaks[:]
        report["elapsed_seconds"] = time.monotonic() - start
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    thread = threading.Thread(target=monitor)
    thread.start()
    gen = None
    original_forward = runtime.model.forward
    target_end = []

    def forward(input_ids, params=None):
        if params is not None and params.get("cache_seqlens") is not None:
            target_end.append(int(params["cache_seqlens"].max()) + input_ids.shape[-1])
        return original_forward(input_ids, params)

    try:
        report["target_cache"] = runtime.model.tp_worker_dispatch_wait_multi(
            runtime.model.active_devices, initialize_rank_cache, (id(runtime.cache),))
        draft_layers = []
        with torch.cuda.device(runtime.draft_model.output_device):
            for layer in runtime.draft_cache.layers.values():
                for tensor in layer.get_tensors():
                    tensor.zero_()
                layer.sk.fill_(.001)
                layer.sv.fill_(.001)
                draft_layers.append({"pages": int(layer.qk.shape[0]),
                                     "tokens": int(layer.shape[0] * layer.shape[1]),
                                     "bytes": int(layer.storage_size())})
        report["draft_cache"] = draft_layers
        save()
        kwargs = runtime.generator_kwargs
        kwargs.update(num_draft_tokens=draft_tokens, dynamic_draft_tokens=False)
        gen = Generator(**kwargs)
        torch.manual_seed(20261002)
        # CPU IDs and page metadata cover the full context. Only the final
        # chunk(s) are evaluated, so this does not run a 768Ki-token prefill.
        ids = torch.randint(500, runtime.model.config.vocab_size - 500,
                            (1, prompt_tokens), dtype=torch.long)
        job = Job(ids, max_new_tokens=output_tokens, min_new_tokens=output_tokens,
                  sampler=ArgmaxSampler(), return_logits=True, stop_conditions=[],
                  decode_special_tokens=True)
        runtime.model.forward = forward

        class Adapter:
            generator = gen

            async def close(self):
                self.generator.clear_queue()

        with runtime.power_context(Adapter()):
            gen.enqueue(job)
            gen.iterate_start_jobs([])
            assert job in gen.active_jobs, "Full-context job was not admitted"
            seq = job.sequences[0]
            assert len(seq.allocated_pages) == capacity // 256
            assert len(set(seq.block_index_tensor.flatten().tolist())) == capacity // 256
            begin = (prompt_tokens - 1 - gen.max_chunk_size) // 256 * 256
            seq.kv_position = begin
            job.recurrent_state.position = begin
            for page in seq.allocated_pages[:begin // 256]:
                page.kv_position = 256
            report["allocated_pages"] = len(seq.allocated_pages)
            report["prefill_begin"] = begin
            report["prefill_tokens_evaluated"] = prompt_tokens - 1 - begin
            report["prefill_chunk"] = gen.max_chunk_size
            save()
            while not job.is_prefill_done():
                job.prefill([])
                runtime.model.tp_worker_dispatch_wait_multi(
                    runtime.model.active_devices, tp_run.tp_sync_rank, ())
                assert job.recurrent_state.position == seq.kv_position
                report["stages"].append({"stage": "tail_prefill", "position": seq.kv_position})
                save()
            logits_checked = 0
            while gen.num_remaining_jobs():
                results = gen.iterate()
                for result in results:
                    if result.get("stage") == "error":
                        raise result["error"]
                    logits = result.get("logits")
                    if logits is not None and logits.numel():
                        # The sampler intentionally masks unused vocabulary
                        # padding with -inf before Job returns its logits.
                        valid = logits[..., :runtime.tokenizer.actual_vocab_size]
                        assert torch.isfinite(valid).all(), "Nonfinite synthetic logits in actual vocabulary"
                        logits_checked += valid.numel()
                save()
            report["generated_tokens"] = job.new_tokens
            report["finite_logits_elements"] = logits_checked
            report["target_max_end_exclusive"] = max(target_end)
            report["accepted_draft_tokens"] = job.accepted_draft_tokens
            report["rejected_draft_tokens"] = job.rejected_draft_tokens
            assert job.new_tokens == output_tokens
            assert max(target_end) <= capacity
            assert max(target_end) >= capacity - draft_tokens - 3
        report["ranks_after"] = runtime.model.tp_worker_dispatch_wait_multi(
            runtime.model.active_devices, tp_run.tp_audit_rank, ())
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        runtime.model.forward = original_forward
        if gen is not None:
            gen.clear_queue()
        done.set()
        thread.join()
        report["samples"] = samples
        save()
    return report
