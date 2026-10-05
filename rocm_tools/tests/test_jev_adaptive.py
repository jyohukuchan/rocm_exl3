from contextlib import contextmanager
import math
from types import SimpleNamespace
import pytest
import torch
from rocm_tools.jev_runtime import JEVRuntime


def runtime(p1):
    r=JEVRuntime.__new__(JEVRuntime);r.context=2048;r.torch=torch
    r.decide=lambda *a,**kw:{'probabilities':list(p1),'options':['false','true'],'choice_index':1,'choice':'true',
                            'usage':{'prompt_tokens':5,'completion_tokens':0,'total_tokens':5},'num_model_requests':1}
    r.state_parts=lambda s:('state',[])
    def encode(text,**kw):
        ids={'Answer: (':[1,2],'Answer: (A)':[1,2,10,3],'Answer: (B)':[1,2,11,3],'</think>':[7]}
        return torch.tensor(ids.get(text,[99]))
    r.tokenizer=SimpleNamespace(encode=encode)
    r.hf_tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:'prompt<think>')
    r.generate_prompt=lambda *a,**kw:{'text':'reason','finish_reason':'length',
                                      'usage':{'prompt_tokens':7,'completion_tokens':3}}
    @contextmanager
    def session(*a,**kw):
        logits=torch.zeros(12);logits[10]=math.log(3)
        yield logits,9,None
    r.session=session
    return r


def test_noul_reorders_yes_no_and_mixes_without_overconfidence():
    r=runtime([0.4,0.6])
    out=r.adaptive_decide('noul','x','q',thinking='auto',threshold=0.8,think_budget=8,debug=True,return_reasoning=True)
    assert out['probabilities']==pytest.approx([0.325,0.675])
    assert out['thinking']['system2']==pytest.approx([0.25,0.75])
    assert out['thinking']['finished_within_budget'] is False
    assert out['thinking']['reasoning']=='reason'
    assert out['usage']=={'prompt_tokens':21,'completion_tokens':3,'total_tokens':24}
    assert out['num_model_requests']==3


def test_high_confidence_does_not_invoke_system2():
    r=runtime([0.01,0.99])
    r.state_parts=lambda *a:(_ for _ in ()).throw(AssertionError('System 2 must be skipped'))
    out=r.adaptive_decide('noul','x','q',thinking='auto')
    assert out['thinking']['used'] is False
    assert out['probabilities']==[0.01,0.99]
    assert out['num_model_requests']==1
    with pytest.raises(ValueError):r.adaptive_decide('score','x','q',thinking='on')


def test_session_keeps_cache_clearing_and_request_in_inference_mode():
    r=JEVRuntime.__new__(JEVRuntime);r.torch=torch
    @contextmanager
    def session(*a,**kw):
        assert torch.is_inference_mode_enabled()
        state=torch.zeros(1)
        yield state
        state.zero_()  # cache-state release/clearing happens before guard exits
    r._session=session
    before=torch.is_inference_mode_enabled()
    with r.session('prompt') as state:
        assert state.is_inference()
        state.add_(1)
        assert state.item()==1
    assert torch.is_inference_mode_enabled()==before
