# SPDX-License-Identifier: MIT
"""Check one-transfer layer metadata packing independently of GPU weight values."""
import json
from pathlib import Path

import torch
from exllamav3.modules.moe_batch_recon import BatchReconLayer

records=[]
for device in (0,1):
    with torch.cuda.device(device):
        layer=BatchReconLayer.__new__(BatchReconLayer)
        layer.device=torch.device('cuda',device)
        tables=tuple(torch.arange(4,device='cuda',dtype=torch.long)*16+base for base in (1000,2000,3000))
        layer.ptr_tables=tables
        meta,ptrs=layer.prepare_groups_metadata([[3,0],[2]],[129,0,193,157,21])
        assert torch.equal(meta.cpu(),torch.tensor([[3,0,2],[322,0,129],[157,129,193]]))
        for base,p in zip((1000,2000,3000),ptrs):assert torch.equal(p.cpu(),torch.tensor([base+48,base,base+32]))
        group=meta[:,1:3]
        assert group.stride(0)==3 and torch.equal(group.cpu(),torch.tensor([[0,2],[0,129],[129,193]]))
        records.append({'device':device,'metadata_and_pointer_alignment_exact':True})
Path('/work/runs/context-batch/moe-metadata-check.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
