# SPDX-License-Identifier: MIT
"""Teacher-force baseline continuations and capture unmodified pre-sampler logits.

Validation only: forced outputs are never reported as unconstrained agreement or
used for throughput. Every literal prompt is checked before recording its rows.
"""

import json
import os
import sys
from pathlib import Path

import torch
from exllamav3.generator.job import Job
from rocm_tools.rdna2 import moe_prefill_probe as probe

_REFERENCE = json.loads(Path(os.environ["EXL3_MOE_QUALITY_REFERENCE"]).read_text())
_PROMPTS = json.loads(Path(sys.argv[sys.argv.index("--prompts-json") + 1]).read_text())[
    "prompts"
]
for prompt, reference in zip(_PROMPTS, _REFERENCE["runs"]):
    assert len(prompt["ids"]) == 1 and isinstance(prompt["ids"][0], list)
    assert prompt["sha"] == reference["ids_sha256"]
    prompt["ids"] = prompt["ids"][0]
_CASES = []
_original_logits = Job.receive_logits


def receive_logits(self, logits):
    assert logits.shape[:2] == (1, 1), "quality probe supports AR batch1 only"
    if not hasattr(self, "_moe_quality_case"):
        case = len(_CASES)
        expected = _PROMPTS[case]["ids"]
        actual = self.sequences[0].sequence_ids.torch_slice(0, len(expected)).cpu()
        assert torch.equal(
            actual, torch.tensor([expected], dtype=torch.long)
        ), "teacher-forcing prompt alignment failed"
        self._moe_quality_case = case
        self._moe_quality_position = 0
        _CASES.append(
            {
                "ids_sha256": _REFERENCE["runs"][case]["ids_sha256"],
                "logits": [],
                "targets": [],
            }
        )
    case, position = self._moe_quality_case, self._moe_quality_position
    vocab = self.generator.tokenizer.actual_vocab_size
    row = logits[0, 0, :vocab].detach().cpu().clone()
    assert torch.isfinite(row).all(), "nonfinite unmasked valid-vocab logit"
    target = _REFERENCE["runs"][case]["tokens"][position]
    _CASES[case]["logits"].append(row)
    _CASES[case]["targets"].append(target)
    self._moe_quality_position += 1
    sampled = _original_logits(self, logits)
    return (torch.full_like(sampled[0], target), *sampled[1:])


Job.receive_logits = receive_logits


def main():
    assert sys.argv[sys.argv.index("--mode") + 1] == "ar"
    status = probe.main()
    assert len(_CASES) == len(_REFERENCE["runs"]) == len(_PROMPTS)
    for case, reference in zip(_CASES, _REFERENCE["runs"]):
        assert case["targets"] == reference["tokens"]
        case["logits"] = torch.stack(case["logits"])
    output = probe._OUTPUT.with_name(probe._OUTPUT.stem + "-logits.pt")
    torch.save(_CASES, output)
    report = json.loads(probe._OUTPUT.read_text())
    report.update(
        validation_only=True,
        teacher_forced_quality=True,
        quality_logits_file=str(output),
    )
    probe._OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
