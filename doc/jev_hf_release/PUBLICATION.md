# JEV Hugging Face release preparation

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
in the artifact. Login was verified for `jyohukuchan`; write permission has not
been tested by a mutation, and no model repository has been created.

After the user specifies the repository and public/private visibility, upload
using the existing local Hugging Face login. Example for the proposed **public**
repository (only after publication is authorized):

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

Use `private=True` if that is the selected visibility. Check the remote file
inventory and LFS/Xet digests against the manifest after upload, then return
the actual repository URL and revision. No hosted inference or general engine
compatibility follows from hosting the weight files.

Primary references:
[Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload),
[model-card metadata](https://huggingface.co/docs/hub/model-cards),
[source model](https://huggingface.co/autotrust/JEV-27B-VL),
[Apache-2.0 redistribution terms](https://www.apache.org/licenses/LICENSE-2.0).
