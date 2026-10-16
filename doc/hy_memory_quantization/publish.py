"""Package a validated Hy-Memory quant and optionally publish it under Apache-2.0."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from huggingface_hub import HfApi, ModelCard


SOURCE = "tencent/Hy-Memory-30B-A3B"
REVISION = "64a460a94d5498b7316fb02395c72d01c47a7347"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="jyohukuchan/Hy-Memory-30B-A3B-exl3-4.08bpw")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    # The conversion container runs as root. Hand only this release directory
    # back to the invoking user before writing its card or creating hard links.
    work = Path("/home/homelab1/datapool/rocm-exl3-rdna2")
    model_dir = args.model_dir.resolve()
    if model_dir.stat().st_uid != os.getuid():
        relative = model_dir.relative_to(work / "models")
        assert relative == Path("Hy-Memory-30B-A3B-exl3-4.08bpw")
        subprocess.run([
            "docker", "exec", "rocm-exl3-r9700-conv", "chown", "-R",
            f"{os.getuid()}:{os.getgid()}", str(Path("/work/models") / relative),
        ], check=True)
    report = json.loads((args.run_dir / "validation.json").read_text())
    assert report["passed"] and len(report["samples"]) == 3
    for shard in report["shards"]:
        path = args.model_dir / shard["file"]
        assert path.stat().st_size == shard["size_bytes"] and sha256(path) == shard["sha256"]
    license_bytes = (args.source_dir / "LICENSE").read_bytes()
    assert b"Apache-2.0" in license_bytes and b"additional restrictions" in license_bytes
    (args.model_dir / "LICENSE").write_bytes(license_bytes)
    shutil.copy2(args.source_dir / "README.md", args.model_dir / "SOURCE_MODEL_CARD.md")
    for name in ("recipe-4bit-base.json", "calibration-metadata.json", "validation.json"):
        shutil.copy2(args.run_dir / name, args.model_dir / name)
    if (args.run_dir / "validation-initial.json").exists():
        shutil.copy2(args.run_dir / "validation-initial.json", args.model_dir / "validation-initial.json")
    packaging_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    revision_file = args.run_dir / "conversion-engine-revision.txt"
    engine_revision = revision_file.read_text().strip() if revision_file.exists() else packaging_revision
    weight_gb = report["total_weight_bytes"] / 1e9
    weight_gib = report["total_weight_bytes"] / 2**30
    card = f'''---
license: apache-2.0
base_model: {SOURCE}
base_model_relation: quantized
language:
  - en
  - zh
pipeline_tag: text-generation
tags:
  - exl3
  - quantized
  - moe
  - memory
---

# Hy-Memory-30B-A3B — EXL3 4.08 bpw

Mixed-precision EXL3 conversion of [{SOURCE}](https://huggingface.co/{SOURCE})
at revision `{REVISION}`. The source is a memory-writer model for extracting,
updating and reconciling persistent information in external memory frameworks,
including Mem0, MemOS and Hy-Memory. Consult the source model card for its intended
use and original evaluation; those scores have not been reproduced for this quant.

## Precision allocation

| Component | Precision |
| --- | --- |
| Routed experts, layers 1–47, all gate/up/down projections | 4 bit |
| Attention, layers 0–47, all Q/K/V/O projections | 6 bit |
| Shared experts, layers 1–47, all gate/up/down projections | 6 bit |
| Dense FFN, layer 0, all gate/up/down projections | 5 bit |
| Output head | 8 bit |
| Input embeddings, routers and norms | Unquantized 16 bit |
| Expert routing correction bias | Original FP32 |

The decoder's parameter-weighted bitrate is **{report['decoder_bpw']:.8f} bpw**;
the name rounds this to 4.08. This excludes the head, embeddings, routers, norms,
and quantization auxiliary tensors. Total stored weight files occupy
**{weight_gb:.3f} GB / {weight_gib:.3f} GiB**. KV cache and runtime buffers require
additional memory. `recipe-4bit-base.json` records all 18,384 decoder allocations.
This is a 4-bit-base allocation with selected precision increases, not a uniform
4-bit quant or a claim of a sensitivity-optimized allocation.

## Conversion and validation

Converted with [rocm_exl3_forme](https://github.com/jyohukuchan/rocm_exl3_forme),
engine revision `{engine_revision}`, on a Radeon AI PRO R9700. Uses the `mul1`
codebook, output channel scales, FP32 Hessians and 250 × 2048 calibration tokens:
150 rows of the bundled general corpus mix and 100 rows of reproducible fictional
memory-writing examples. The latter cover extraction, fact correction, exact coding
details and Japanese text. They are synthetic examples, not private user data or
a self-sampled domain trace. See `calibration-metadata.json` for the token-file digest.

All stored decoder bitrates and the head were checked against the recipe.
Three held-out functional smoke checks passed: explicit fact extraction,
replacement of a superseded location while preserving a dietary preference,
and abstention when employer/address evidence is absent. Full prompts and responses,
GPU/runtime details and weight SHA-256 digests are in `validation.json`.
These checks do not measure quality loss relative to BF16 or establish benchmark
equivalence with the source model.

**Prompt specificity:** in an initial check with broader instructions, the model
kept a superseded location with an annotation and retained a temporary weather
observation. The final checks explicitly request only currently valid facts,
complete removal of superseded items and exclusion of transient weather.
`validation-initial.json` preserves those original prompts, responses and the
failed correction check. Use explicit memory-bank policies in deployment;
these observations have not been compared against BF16 and cannot be attributed
specifically to quantization.

## Download and inference

```bash
hf download {args.repo_id} --local-dir ./Hy-Memory-EXL3
```

Use a HYV3-compatible ExLlamaV3 runtime. In the ROCm fork, revision `67cbdd1`
or later fixes the expert-width interpretation required by this checkpoint:
the actual width is `expert_hidden_dim=768`, although the original config also
contains the legacy `moe_intermediate_size=1536`. Keep the published config intact.

```python
from exllamav3 import Cache, Config, Generator, Model, Tokenizer, TopPSampler

config = Config.from_directory("./Hy-Memory-EXL3")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096, max_batch_size=1)
model.load(device="cuda:0")
tokenizer = Tokenizer.from_config(config)
generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
prompt = tokenizer.hf_render_chat_template([
    {{"role": "system", "content": 'Extract explicit user facts. Do not infer missing information. Return only JSON with a "memories" list of strings.'}},
    {{"role": "user", "content": "My name is Nora. I live in Madrid and am vegetarian."}},
], reasoning_effort="no_think")
print(generator.generate(
    prompt, max_new_tokens=256, completion_only=True,
    encode_special_tokens=True, stop_conditions=config.eos_token_id_list,
    sampler=TopPSampler(top_p=1.0, temperature=0.0),
))
```

The native `chat_template.jinja` and tokenizer files are included. This EXL3
weight format requires ExLlamaV3; ordinary Transformers weight loading does
not decode the EXL3 trellis tensors. The source model supports a 262,144-token
position limit; long-context behavior and that cache size were not validated here.

## License and attribution

Apache-2.0, matching the source weights with no additional restrictions.
The original Tencent copyright and full license are preserved in `LICENSE`.
`NOTICE` identifies this quantization as a modified distribution;
`SOURCE_MODEL_CARD.md` preserves the original model-card text. All quality
claims and provenance for the original model belong to its authors.
'''
    (args.model_dir / "README.md").write_text(card)
    (args.model_dir / "NOTICE").write_text(
        "Hy-Memory-30B-A3B\nCopyright (C) 2026 Tencent. All rights reserved.\n\n"
        f"Source: https://huggingface.co/{SOURCE}\nSource revision: {REVISION}\n"
        "This distribution modifies the original weights by mixed-precision EXL3 quantization.\n"
        "Quantization and release packaging by jyohukuchan. Original weights are Apache-2.0.\n"
    )
    metadata = {"source_model": SOURCE, "source_revision": REVISION,
                "engine_revision": engine_revision, "packaging_revision": packaging_revision,
                "license": "apache-2.0",
                "decoder_bpw": report["decoder_bpw"], "head_bits": 8,
                "repo_id": args.repo_id, "created_at": datetime.now(timezone.utc).isoformat()}
    (args.model_dir / "RELEASE_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
    parsed = ModelCard.load(str(args.model_dir / "README.md"))
    assert parsed.data.license == "apache-2.0" and parsed.data.base_model_relation == "quantized"
    files = [p for p in sorted(args.model_dir.iterdir()) if p.is_file() and p.name != "FILE_MANIFEST.json"]
    entries = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    (args.model_dir / "FILE_MANIFEST.json").write_text(json.dumps({"files": entries}, indent=2) + "\n")
    if not args.publish:
        print(json.dumps({"prepared": str(args.model_dir), "repo_id": args.repo_id}))
        return
    api = HfApi()
    assert api.whoami()["name"] == args.repo_id.split("/")[0]
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=False)
    commit = api.upload_folder(
        repo_id=args.repo_id, folder_path=str(args.model_dir),
        allow_patterns=[p.name for p in files] + ["FILE_MANIFEST.json"],
        commit_message="Publish validated mixed-precision EXL3 quantization of Hy-Memory-30B-A3B",
    )
    # Verify anonymous metadata and every publicly exposed LFS digest/size.
    remote = HfApi(token=False).model_info(args.repo_id, revision=commit.oid, files_metadata=True)
    assert not remote.private and remote.card_data.license == "apache-2.0"
    by_name = {f.rfilename: f for f in remote.siblings}
    assert set(by_name) == {p.name for p in files} | {"FILE_MANIFEST.json", ".gitattributes"}
    for entry in entries:
        info = by_name[entry["path"]]
        assert info.size == entry["bytes"], entry["path"]
        if entry["path"].endswith(".safetensors"):
            assert info.lfs.sha256 == entry["sha256"], entry["path"]
    result = {"repo_id": args.repo_id, "url": f"https://huggingface.co/{args.repo_id}",
              "commit": commit.oid, "commit_url": commit.commit_url,
              "license": "apache-2.0", "uploaded_at": datetime.now(timezone.utc).isoformat(),
              "files": sorted(by_name), "verification": "Anonymous metadata; all sizes and weight LFS SHA-256 digests match."}
    (args.run_dir / "hf-publication.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
