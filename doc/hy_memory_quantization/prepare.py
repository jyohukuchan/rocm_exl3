"""Prepare the pinned HYV3 allocation and a reproducible mixed calibration set."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch
from safetensors.torch import save_file

from exllamav3 import Config, Model, Tokenizer
from exllamav3.conversion.allocation import create_q_strategy
from exllamav3.conversion.calibration_data import get_default_calibration
from exllamav3.modules import Linear


SOURCE_REVISION = "64a460a94d5498b7316fb02395c72d01c47a7347"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def memory_episode(index, rng):
    """Fictional calibration examples; never taken from a user's memories."""
    names = ["Alice", "Diego", "Mei", "Sam", "Priya", "Yuki", "Omar", "Lina"]
    cities = ["Osaka", "Berlin", "Taipei", "Lima", "Lyon", "Kyoto", "Boston", "Seoul"]
    name = rng.choice(names)
    old, new = rng.sample(cities, 2)
    language = rng.choice(["Python", "Rust", "Go", "TypeScript", "C++", "Java"])
    number = rng.randrange(100, 9999)
    mode = index % 4
    if mode == 0:
        system = ('Extract only explicit stable user facts. Do not infer missing facts. '
                  'Return a JSON object with a "memories" list of strings.')
        user = (f'My name is {name}. I live in {new}. I use {language} at work. '
                'I prefer concise answers. The weather is cloudy today.')
        answer = {"memories": [f"The user's name is {name}.", f"The user lives in {new}.",
                               f"The user uses {language} at work.", "The user prefers concise answers."]}
    elif mode == 1:
        system = ('Update the memory bank using explicit new evidence. Replace superseded facts, '
                  'preserve unrelated facts, and return JSON with a "memories" list of strings.')
        user = (f'Existing bank: ["{name} lives in {old}", "{name} is vegetarian"]. '
                f'New message: I moved to {new} permanently. I still avoid meat.')
        answer = {"memories": [f"{name} lives in {new}.", f"{name} is vegetarian."]}
    elif mode == 2:
        system = ('Extract explicit persistent coding decisions. Preserve exact paths, versions, '
                  'numbers and negations. Output JSON with a "memories" list of strings.')
        user = (f'Repository project-{number}: use {language}. In src/cache_{number}.py the '
                f'cache limit is now {number} entries, replacing 64. Never log access tokens. '
                'The previous debug print was temporary and has been removed.')
        answer = {"memories": [f"project-{number} uses {language}.",
                               f"src/cache_{number}.py has a cache limit of {number} entries.",
                               "Never log access tokens."]}
    else:
        system = ('明示された長期的なユーザー情報だけを抽出してください。推測はしないでください。'
                  'memoriesという文字列配列を持つJSONだけを返してください。')
        user = (f'私は{name}です。仕事では{language}を使います。回答は日本語で短くお願いします。'
                '今日は電車が遅れていました。会社名や住所はまだ伝えていません。')
        answer = {"memories": [f"ユーザーの名前は{name}。", f"仕事で{language}を使う。",
                               "短い日本語の回答を好む。"]}
    return [{"role": "system", "content": system}, {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-shapes", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = Config.from_directory(str(args.model_dir))
    assert config.moe_intermediate_size == 768
    model = Model.from_config(config)
    if args.check_shapes:
        for module in model:
            if not isinstance(module, Linear):
                continue
            meta = config.stc.get_tensor_meta(module.key + ".weight", optional=False)
            shape = meta[module.key + ".weight"]["shape"]
            expected = [module.out_features_unpadded, module.in_features_unpadded]
            assert shape == expected, (module.key, shape, expected)
    strategy, achieved = create_q_strategy(model, None, config, 4.0, 8, 4, True)
    head_bits = strategy.pop("lm_head")
    counts = dict(Counter(strategy.values()))
    assert counts == {4: 18048, 5: 3, 6: 333}, counts
    assert abs(achieved - 4.077745849297574) < 1e-12
    recipe = {"source_model": "tencent/Hy-Memory-30B-A3B", "source_revision": SOURCE_REVISION,
              "target_bpw": 4.0, "achieved_bpw": achieved, "head_bits": head_bits,
              "tensor_counts": counts, "tensors": strategy}
    write_json(args.output / "recipe-4bit-base.json", recipe)

    tokenizer = Tokenizer.from_config(config)
    rows = get_default_calibration({"cal_rows": 150, "cal_cols": 2048}, tokenizer)
    rng = random.Random(20261016)
    episode = 0
    for _ in range(100):
        tokens = []
        while len(tokens) < 2048:
            messages = memory_episode(episode, rng)
            rendered = tokenizer.hf_render_chat_template(
                messages, add_generation_prompt=False, reasoning_effort="no_think", is_training=True,
            )
            tokens.extend(tokenizer.encode(rendered, add_bos=False, encode_special_tokens=True)[0].tolist())
            episode += 1
        rows.append(torch.tensor([tokens[:2048]], dtype=torch.long))
    # Interleave corpus and task examples rather than place all task rows last.
    random.Random(20261016).shuffle(rows)
    packed = torch.cat(rows)
    assert packed.shape == (250, 2048)
    path = args.output / "calibration.safetensors"
    save_file({"input_ids": packed}, str(path))
    write_json(args.output / "calibration-metadata.json", {
        "rows": 250, "columns": 2048, "seed": 20261016,
        "general_corpus_rows": 150, "synthetic_memory_rows": 100,
        "synthetic_episodes": episode, "source_revision": SOURCE_REVISION,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "description": "Bundled EXL3 corpus mix plus fictional extraction, correction, coding and Japanese memory examples; not a self-sampled trace or benchmark dataset.",
    })
    config.stc.close()
    print(json.dumps({"recipe": str(args.output / "recipe-4bit-base.json"),
                      "achieved_bpw": achieved, "counts": counts, "calibration_shape": list(packed.shape)}))


if __name__ == "__main__":
    main()
