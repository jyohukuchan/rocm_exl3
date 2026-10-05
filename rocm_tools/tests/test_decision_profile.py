import math
import pytest
import torch
from exllamav3.model.decision import DecisionProfile
from exllamav3.modules.linear import Linear


def profile():
    labels = list('ABCDEFGHIJKLMNOP')
    head = {'bias':[0.2,-0.2]+[0.0]*22,
            'verbalizer_ids':[3721,1802]+list(range(15,21))+list(range(32,48)),
            'slots':{'num_slots':24,'template_version':'bare-v1',
                     'ranges':{'noul':[0,2],'score':[2,8],'choice':[8,24]}}}
    def encode(s):
        if s.startswith('x\n'):
            s = s[2:].split(')')[0]
        if s in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            return [32+ord(s)-ord('A')]
        if len(s)==2 and s.isupper():
            return [1000+26*(ord(s[0])-65)+ord(s[1])-65]
        return [0,1]
    return DecisionProfile(head,{'per_kind':{'noul':2.0,'score':1.0,'choice':1.0}},encode)


def test_bias_before_temperature_and_no_sampling_truncation():
    p = profile()
    q = p.question('noul','Is it correct?')
    result = p.probabilities([0.0,2.0],q)
    assert result[1] == pytest.approx(1/(1+math.exp(-0.8)))
    assert sum(result)==pytest.approx(1)
    assert q['suffix']=='\n[question] Is it correct?\n[options]\nfalse\ntrue\n[decision]:'


def test_extended_labels_and_zero_bias_after_trained_slots():
    p = profile()
    q = p.question('choice','Pick', [str(i) for i in range(256)])
    assert len(q['ids'])==256 and q['ids'][26]==1000
    assert q['bias'][16:]==[0.0]*240
    assert len(p.probabilities([10000.0]*256,q))==256
    with pytest.raises(ValueError): p.question('choice','Pick',['a']*257)
    with pytest.raises(ValueError): p.probabilities([float('nan')]*256,q)


def test_source_head_override_bypasses_quantized_head_and_head_lora():
    m = Linear(None,'lm_head',8,8,caps={'logits_output':True},pad_to=8)
    m.inner = None  # cannot be called: override does not reconstruct vocabulary head
    x = torch.randn(1,2,8).half()
    exact = torch.randn(8,3)
    actual = m.forward(x,{'head_override':exact})
    torch.testing.assert_close(actual,x.float()@exact)
    with pytest.raises(ValueError):m.forward(x,{'head_override':torch.zeros(7,3)})
