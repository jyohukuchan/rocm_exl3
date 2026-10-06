# JEV Hugging Face release preparation

Published on 2026-10-06:
[jyohukuchan/JEV-27B-VL-exl3-4bpw](https://huggingface.co/jyohukuchan/JEV-27B-VL-exl3-4bpw),
revision `9a811e2f521a4d2436da23937a056f299f0cfd3d`.
All 44 release files match the local manifest by size and LFS/Xet SHA256 or
Git blob digest; all 19 safetensors are verified. Downloaded README/manifest/
metadata also match, and public metadata is readable without authentication.
See [hf-verification.json](hf-verification.json).

The model card, attribution notice and launcher in this directory are templates
for the staged EXL3 model release. The preparation command performs no Hub
repository creation or upload and leaves the validated source model unchanged:

```sh
python -m rocm_tools.prepare_jev_hf_release \
  --model /home/homelab1/datapool/ai_models/safetensors/JEV-27B-VL-exl3-4bpw \
  --output /home/homelab1/datapool/rocm-exl3-rdna2/releases/JEV-27B-VL-exl3-4bpw \
  --repo-id jyohukuchan/JEV-27B-VL-exl3-4bpw
```

The current staged release has 44 files, 17,445,032,892 bytes (16.247 GiB),
and 19 hardlinked safetensors files. All 36 retained original files have the
same SHA256 as the validated pack. The source README is archived as
`SOURCE_MODEL_CARD.md`; the new card names the EXL3 format and required ROCm
engine, removes the source Transformers-library claim, and describes only this
release's verified quality/latency results. Source vLLM serving scripts are not
part of the staged release. Weight files must not be modified in place because
their inodes are shared with the validated model.

`FILE_MANIFEST.json` records all staged file SHA256 values except its own.
Shard headers/index, model-card YAML and launcher syntax were checked.
Model/adapter assets retain Apache-2.0 with LICENSE/NOTICE/source attribution;
the launcher has a separate MIT code license. The API token is never included
in the artifact. The initial token could read but could not create the model
repository (403). After the user authorized publication and switched the local
login to a write-capable token, creation and upload succeeded under `jyohukuchan`.

The original public publication used the existing local Hugging Face login:

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "jyohukuchan/JEV-27B-VL-exl3-4bpw"
api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=False)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path="/home/homelab1/datapool/rocm-exl3-rdna2/releases/JEV-27B-VL-exl3-4bpw",
    commit_message="Add validated JEV-27B-VL EXL3 mixed-precision ROCm release",
)
```

The repository now exists; the above creation step records the original
publication, rather than a command needed to download the model. No hosted
inference or general engine compatibility follows from hosting the weight files.

Primary references:
[Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload),
[model-card metadata](https://huggingface.co/docs/hub/model-cards),
[source model](https://huggingface.co/autotrust/JEV-27B-VL),
[Apache-2.0 redistribution terms](https://www.apache.org/licenses/LICENSE-2.0).
