import hashlib
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from safetensors import safe_open
from rocm_tools.prepare_qwen_image21_hf_release import annotate_tensor_file


def test_publication_preserves_tensors_and_source(tmp_path):
    src, dst = tmp_path/'original.safetensors', tmp_path/'published.safetensors'
    values={'trellis':np.arange(48,dtype=np.int16).reshape(2,3,8), 'scale':np.array([0.1,0.9],dtype=np.float16)}
    save_file(values,src,metadata={'existing':'keep'})
    before=src.read_bytes(); digest=hashlib.sha256(before).hexdigest()
    info=annotate_tensor_file(src,dst,digest,{'MODIFICATION_NOTICE':'Changed for publication','license':'research'})
    assert src.read_bytes()==before
    assert info['sha256']==hashlib.sha256(dst.read_bytes()).hexdigest()!=digest
    assert info['tensor_payload_identical']
    for key,value in load_file(dst).items():np.testing.assert_array_equal(value,values[key])
    with safe_open(dst,framework='np') as f:
        assert f.metadata()['existing']=='keep'
        assert f.metadata()['MODIFICATION_NOTICE']=='Changed for publication'


def test_bad_source_checksum_cannot_publish(tmp_path):
    src,dst=tmp_path/'source.safetensors',tmp_path/'output.safetensors'
    save_file({'x':np.ones(2,dtype=np.float32)},src)
    with pytest.raises(ValueError,match='checksum'):
        annotate_tensor_file(src,dst,'bad',{'MODIFICATION_NOTICE':'Changed'})
    assert not dst.exists() and not dst.with_suffix('.safetensors.tmp').exists()
