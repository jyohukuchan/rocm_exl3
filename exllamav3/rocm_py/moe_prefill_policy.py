# SPDX-License-Identifier: MIT
"""Pure selection rules for the measured V620/Qwen3.8 TP2 prefill geometry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillPolicy:
    enabled: bool = False
    fused_rows: int = 128
    fusion: str = "0"


def select_policy(
    *,
    arch,
    geometry,
    projections,
    triton_compatible=False,
    folded=False,
    deterministic=True
):
    """Other architectures, formats and placements keep their original route."""
    eligible = (
        arch.split(":", 1)[0] == "gfx1030"
        and geometry == (2560, 640, 512, 256, 10, "silu", True)
        and projections
        == (
            (2560, 640, 3, False, True),
            (2560, 640, 3, False, True),
            (640, 2560, 3, False, True),
        )
        and not folded
        and deterministic
    )
    if not eligible:
        return PrefillPolicy()
    return PrefillPolicy(True, 32, "fused" if triton_compatible else "0")
