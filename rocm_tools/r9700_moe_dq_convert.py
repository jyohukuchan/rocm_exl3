import runpy

# R9700 (gfx1201) conversion workaround: RDNA4 mma_sync traps in the cooperative
# BC graph / fused MoE kernels (see doc/r9700_vs_v620.md and the inference-side
# adapter r9700_moe_dq_run.py). The conversion's state-advance forwards run the
# quantized module at bsz = 1, which lands on exactly those kernels. Zeroing
# TEMP_ROWS_GRAPH and MAX_BSZN only for the duration of each BlockSparseMLP
# forward steers every expert through the per-expert DQ (dequant + GEMM) path,
# which is proven on this GPU; load-time buffer sizing and the
# num_experts_per_tok assert are untouched because the values are restored in
# finally.
from exllamav3.modules import block_sparse_mlp as bs

_orig = bs.BlockSparseMLP.forward

def forward_dq(self, x, *a, **k):
    prev_graph = bs.TEMP_ROWS_GRAPH
    prev_bszn = bs.MAX_BSZN
    bs.TEMP_ROWS_GRAPH = 0
    bs.MAX_BSZN = 0
    try:
        return _orig(self, x, *a, **k)
    finally:
        bs.TEMP_ROWS_GRAPH = prev_graph
        bs.MAX_BSZN = prev_bszn

bs.BlockSparseMLP.forward = forward_dq
print("[MOE-DQ] all MoE forwards steered to the per-expert DQ path (R9700 workaround)", flush = True)

runpy.run_module("exllamav3.conversion.convert_model", run_name = "__main__")
