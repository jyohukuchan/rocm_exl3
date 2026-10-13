"""Check the reference-derived allocation and run two short translation samples."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from exllamav3 import Cache, Config, Generator, Model, Tokenizer, TopPSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    recipe = json.loads(Path(__file__).with_name("recipe-4.05bpw.json").read_text())

    assert torch.cuda.device_count() == 1, "Expose only the R9700"
    device_name = torch.cuda.get_device_name(0)
    assert "R9700" in device_name, device_name
    config = Config.from_directory(str(args.model_dir))
    model = Model.from_config(config)
    # The converter's storage manifest describes the text model; examine its actual
    # safetensors shapes independently, so a stale manifest cannot hide wrong bits.
    actual = {}
    numel = total_bits = 0

    def tensor_shape(key):
        return config.stc.get_tensor_meta(key, optional=False)[key]["shape"]

    decoder_quantized = {
        key.removesuffix(".trellis") for key in config.stc.tensor_file_map
        if key.startswith("model.language_model.layers.") and key.endswith(".trellis")
    }
    assert decoder_quantized == set(recipe["tensors"])
    for key, expected_bits in recipe["tensors"].items():
        actual[key] = tensor_shape(key + ".trellis")[-1] // 16
        assert actual[key] == expected_bits, (key, actual[key], expected_bits)
        n = tensor_shape(key + ".suh")[0] * tensor_shape(key + ".svh")[0]
        numel += n
        total_bits += n * actual[key]
    assert tensor_shape("lm_head.trellis")[-1] // 16 == 8
    qcfg = json.loads((args.model_dir / "config.json").read_text())["quantization_config"]
    assert qcfg["bits"] == 4.05 and qcfg["head_bits"] == 8
    assert qcfg["mtp_bits"] == 4
    assert qcfg["out_scales"] == "auto" and qcfg["codebook"] == "mul1"
    assert abs(total_bits / numel - recipe["achieved_bpw"]) < 1e-10
    mtp_bits = {key: tensor_shape(key)[-1] // 16
                for key in config.stc.tensor_file_map
                if key.startswith("mtp.") and key.endswith(".trellis")}
    assert len(mtp_bits) == 8 and set(mtp_bits.values()) == {4}, mtp_bits
    config.stc.close()

    cache = Cache(model, max_num_tokens=2048, max_batch_size=1)
    model.load(device="cuda:0", progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    samples = [
        ("en-ja", "请将以下英语文本翻译为日语，直接输出翻译结果，不要进行任何解释。\n\n"
         "The meeting will start at ten tomorrow morning. Please bring the latest report."),
        ("ja-en", "请将以下日语文本翻译为英语，直接输出翻译结果，不要进行任何解释。\n\n"
         "このモデルは日本語と英語の間で文章を翻訳できます。"),
    ]
    translations = []
    for direction, instruction in samples:
        prompt = tokenizer.hf_render_chat_template(
            [{"role": "user", "content": instruction}], enable_thinking=False,
        )
        start = time.monotonic()
        response = generator.generate(
            prompt, max_new_tokens=128, completion_only=True,
            encode_special_tokens=True, stop_conditions=config.eos_token_id_list,
            sampler=TopPSampler(top_p=1.0, temperature=0.0),
        )
        assert response.strip(), f"Empty response: {direction}"
        translations.append({"direction": direction, "instruction": instruction,
                             "response": response, "wall_s": time.monotonic() - start})
        print(json.dumps(translations[-1], ensure_ascii=False), flush=True)

    shards = []
    for path in sorted(args.model_dir.glob("*.safetensors")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        shards.append({"file": path.name, "size_bytes": path.stat().st_size,
                       "sha256": digest.hexdigest()})
    report = {"gpu": device_name, "visible_devices": torch.cuda.device_count(),
              "torch": torch.__version__, "hip": torch.version.hip,
              "decoder_numel": numel, "decoder_bpw": total_bits / numel,
              "decoder_tensor_bits": dict(Counter(actual.values())),
              "mtp_tensor_bits": dict(Counter(mtp_bits.values())),
              "quantization_config": qcfg, "translations": translations,
              "shards": shards, "total_weight_bytes": sum(s["size_bytes"] for s in shards),
              "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(0)}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
