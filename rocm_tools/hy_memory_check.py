"""Run bounded Hy-Memory generation checks on RDNA2/RDNA4, optionally TP2."""

import argparse
import hashlib
import json
import resource
from pathlib import Path
import time

import torch
from exllamav3 import Cache, Config, Generator, Model, Tokenizer, TopPSampler
from exllamav3.ext import exllamav3_ext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--tp", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # ROCm's many small TP tensor allocations use dma-buf descriptors; the
    # container's default 1024 otherwise surfaces as a misleading GPU OOM.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 65536:
        raise RuntimeError("Set the container/launcher nofile hard limit to at least 65536")
    if soft != resource.RLIM_INFINITY and soft < 65536:
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                "arch": torch.cuda.get_device_properties(i).gcnArchName.split(":")[0]}
               for i in range(torch.cuda.device_count())]
    assert all(d["arch"] in ("gfx1030", "gfx1201") for d in devices)
    if args.tp:
        assert len(devices) == 2 and all(d["arch"] == "gfx1030" for d in devices)
    config = Config.from_directory(args.model)
    assert config.architecture == "HYV3ForCausalLM" and config.moe_intermediate_size == 768
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    prompt = model.default_chat_prompt("ping", "Keep answers concise.")
    assert prompt == tokenizer.hf_render_chat_template([
        {"role": "system", "content": "Keep answers concise."},
        {"role": "user", "content": "ping"}], reasoning_effort="no_think")
    assert ":opensource" not in prompt and prompt.endswith("<think></think>")
    cache = Cache(model, max_num_tokens=8192, max_batch_size=1)
    start = time.monotonic()
    if args.tp:
        model.load(tensor_p=True, use_per_device=[24, 24], tp_backend="nccl",
                   tp_output_device="cuda:1", max_chunk_size=2048)
        assert model.loaded_tp and sorted(model.active_devices) == [0, 1]
    else:
        model.load(device="cuda:0")
    load_s = time.monotonic() - start
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, max_chunk_size=2048)
    tasks = [
        ("extraction", 'Extract persistent facts only; exclude temporary weather. '
         'Return only JSON with a "memories" list of strings.',
         'My name is Nora. I live in Madrid and use Julia at work. I am vegetarian. It is raining today.'),
        ("long_correction", 'Return only current facts in JSON with a "memories" list of strings. '
         'Delete superseded facts completely, including annotations and history. Keep unrelated facts.',
         ('Archived reference note: synthetic log entries are not user facts.\n' * 250)
         + 'Existing bank: ["Nora lives in Rome", "Nora is vegetarian"]. '
         'Latest message from Nora: I moved permanently to Lisbon. I still do not eat meat.'),
        ("abstention_after_long", 'Extract explicitly stated employer and home-address facts only. '
         'Do not infer or guess. Return JSON with a "memories" list; use an empty list if neither is stated.',
         'I enjoy green tea and science fiction. Can you guess my employer and address?'),
    ]
    samples = []
    for name, system, user in tasks:
        prompt = model.default_chat_prompt(user, system)
        input_ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
        start = time.monotonic()
        response = generator.generate(prompt, max_new_tokens=256, completion_only=True,
                                      encode_special_tokens=True, stop_conditions=config.eos_token_id_list,
                                      sampler=TopPSampler(top_p=1, temperature=0))
        data = json.loads(response.strip())
        memories = data["memories"]
        assert isinstance(memories, list) and all(isinstance(m, str) for m in memories)
        text = " ".join(memories).lower()
        if name == "extraction":
            assert all(s in text for s in ("nora", "madrid", "julia", "vegetarian"))
            assert "rain" not in text
        elif name == "long_correction":
            assert "lisbon" in text and "rome" not in text and "vegetarian" in text
        else:
            assert memories == []
        sample = {"task": name, "prompt_tokens": input_ids.numel(), "response": response,
                  "wall_s": time.monotonic() - start, "passed": True}
        samples.append(sample)
        print(json.dumps(sample, ensure_ascii=False), flush=True)
    report = {"passed": True, "devices": devices, "tensor_parallel": args.tp,
              "active_devices": model.active_devices, "load_s": load_s,
              "native_extension": exllamav3_ext.__file__, "torch": torch.__version__,
              "native_extension_sha256": hashlib.sha256(Path(exllamav3_ext.__file__).read_bytes()).hexdigest(),
              "hip": torch.version.hip, "samples": samples,
              "rlimit_nofile": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
              "scope": "Functional generation checks; no BF16 quality or maximum-context claim."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    model.unload()


if __name__ == "__main__":
    main()
