"""Verify every EXL3 allocation and exercise three held-out memory-writing tasks."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from exllamav3 import Cache, Config, Generator, Model, Tokenizer, TopPSampler


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    config = Config.from_directory(str(args.model_dir))
    model = Model.from_config(config)
    actual = {key.removesuffix(".trellis"): config.stc.get_tensor_meta(key)[key]["shape"][-1] // 16
              for key in config.stc.tensor_file_map if key.endswith(".trellis")}
    assert actual == {**recipe["tensors"], "lm_head": 8}, "Stored allocation differs from recipe"
    total_bits = numel = 0
    for key, bits in recipe["tensors"].items():
        module = model.find_module(key)
        n = module.weights_numel()
        total_bits += n * bits
        numel += n
    assert abs(total_bits / numel - recipe["achieved_bpw"]) < 1e-12
    for key in ("model.layers.1.mlp.experts.0.up_proj", "model.layers.47.mlp.shared_mlp.up_proj"):
        assert model.find_module(key).out_features == 768
    qcfg = json.loads((args.model_dir / "config.json").read_text())["quantization_config"]
    assert qcfg["bits"] == 4.08 and qcfg["head_bits"] == 8
    assert qcfg["codebook"] == "mul1"
    unquantized_dtypes = Counter()
    for key in config.stc.tensor_file_map:
        if key == "model.embed_tokens.weight" or key.endswith("norm.weight") or key.endswith("router.gate.weight"):
            dtype = config.stc.get_tensor_meta(key)[key]["dtype"]
            assert dtype in ("F16", "BF16"), (key, dtype)
            unquantized_dtypes[dtype] += 1
        elif key.endswith(".mlp.expert_bias"):
            dtype = config.stc.get_tensor_meta(key)[key]["dtype"]
            assert dtype == "F32", (key, dtype)
            unquantized_dtypes[dtype] += 1
    assert sum(unquantized_dtypes.values()) == 288, unquantized_dtypes
    config.stc.close()
    cache = Cache(model, max_num_tokens=4096, max_batch_size=1)
    model.load(device="cuda:0", progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    tasks = [
        ("extraction", 'Extract explicit stable user facts. Do not infer or invent information. '
         'Return only a JSON object with a "memories" list of strings.',
         'My name is Nora. I live in Madrid. I am vegetarian and use Julia for work. '
         'I prefer concise answers. It is raining today.'),
        ("correction", 'Update the memory bank from the latest explicit evidence. Replace superseded '
         'facts and preserve unrelated facts. Return only JSON with a "memories" list of strings.',
         'Existing bank: ["Nora lives in Rome", "Nora is vegetarian"]. '
         'New message from Nora: I moved permanently to Lisbon. I still do not eat meat.'),
        ("abstention", 'Extract only explicitly stated employer and home-address facts. '
         'Do not infer either from preferences. Return only JSON with a "memories" list of strings. '
         'If neither is stated, return {"memories": []}.',
         'I enjoy green tea and often read science fiction. Can you guess my employer and address?'),
    ]
    samples = []
    failures = []
    for name, system, user in tasks:
        prompt = tokenizer.hf_render_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            reasoning_effort="no_think",
        )
        start = time.monotonic()
        response = generator.generate(
            prompt, max_new_tokens=256, completion_only=True, encode_special_tokens=True,
            stop_conditions=config.eos_token_id_list,
            sampler=TopPSampler(top_p=1.0, temperature=0.0),
        )
        passed = False
        error = None
        try:
            # A fenced JSON object is still valid content; record the original response too.
            content = response.strip()
            if content.startswith("```json") and content.endswith("```"):
                content = content[len("```json"):-3].strip()
            data = json.loads(content)
            memories = data["memories"]
            assert isinstance(memories, list) and all(isinstance(m, str) for m in memories)
            text = " ".join(memories).lower()
            if name == "extraction":
                assert all(word in text for word in ("nora", "madrid", "vegetarian", "julia"))
            elif name == "correction":
                assert "lisbon" in text and "rome" not in text
                assert "vegetarian" in text or "meat" in text
            else:
                assert memories == []
            passed = True
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            failures.append(name)
        sample = {"task": name, "system": system, "user": user, "response": response,
                  "passed": passed, "error": error, "wall_s": time.monotonic() - start}
        samples.append(sample)
        print(json.dumps(sample, ensure_ascii=False), flush=True)
    shards = [{"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
              for path in sorted(args.model_dir.glob("*.safetensors"))]
    report = {"passed": not failures, "failures": failures,
              "scope": "Allocation audit and three functional smoke checks; not a BF16 quality comparison.",
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "hip": torch.version.hip,
              "decoder_numel": numel, "decoder_bpw": total_bits / numel,
              "decoder_tensor_bits": dict(Counter(recipe["tensors"].values())),
              "unquantized_tensor_dtypes": dict(unquantized_dtypes),
              "quantization_config": qcfg, "samples": samples, "shards": shards,
              "total_weight_bytes": sum(s["size_bytes"] for s in shards),
              "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(0)}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"Memory smoke checks failed: {failures}")


if __name__ == "__main__":
    main()
