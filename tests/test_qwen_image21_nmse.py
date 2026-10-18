import pytest
import torch
from rocm_tools.qwen_image21_nmse import OutputNMSE


class Cache:
    def __init__(self):
        self.value = None


def test_nmse_uses_target_tokens_own_cache_and_ratio_of_sums():
    reference_cache = Cache()
    reference_cache.value = 99

    def candidate(**kw):
        cache = kw['kv_cache']
        assert cache is not reference_cache
        if kw['kv_cache_mode'] == 'extract':
            cache.value = 2
            return (torch.tensor([[[999.], [2.], [4.]]]),)
        assert cache.value == 2
        return (torch.tensor([[[3.], [5.]]]),)

    metric = OutputNMSE(candidate, Cache)
    kw = dict(kv_cache=reference_cache, kv_cache_mode='extract',
              timestep=torch.tensor([1.]), img_shapes=[[(1, 1, 2)]])
    ref = (torch.tensor([[[123.], [1.], [2.]]]),)
    assert metric.observe(None, (), kw, ref) is None
    assert ref[0][0, 0, 0] == 123 and reference_cache.value == 99
    metric.step = 1
    metric.observe(None, (), {**kw, 'kv_cache_mode': 'cached'}, (torch.tensor([[[3.], [4.]]]),))
    result = metric.report()
    assert result['nmse'] == pytest.approx(6 / 30)
    assert result['elements'] == 4
    assert [r['step'] for r in result['steps']] == [0, 1]
    with pytest.raises(ValueError, match='without prefix'):
        metric.observe(None, (), {**kw, 'kv_cache': Cache(), 'kv_cache_mode': 'cached'}, ref)


def test_nmse_self_zero_and_rejects_nonfinite():
    kw = dict(timestep=torch.tensor([1.]), img_shapes=[[(1, 1, 2)]])
    ref = (torch.tensor([[[1.], [2.]]], dtype=torch.bfloat16),)
    metric = OutputNMSE(lambda **kw: ref, Cache)
    metric.observe(None, (), kw, ref)
    assert metric.report()['nmse'] == 0
    metric.candidate_forward = lambda **kw: (torch.full((1, 2, 1), float('nan')),)
    with pytest.raises(ValueError, match='Nonfinite'):
        metric.observe(None, (), kw, ref)
