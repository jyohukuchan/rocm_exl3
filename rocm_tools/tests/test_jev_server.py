from types import SimpleNamespace
from fastapi.testclient import TestClient
from rocm_tools.jev_server import create_app,format_answer,systemone_questions
import pytest


class FakeRuntime:
    profile=SimpleNamespace(labels=list(range(256)),temperatures={'noul':1,'score':1,'choice':1})
    context=16384
    def __init__(self):self.calls=[]
    def decide(self,kind,state,question,options=None,**kwargs):
        self.calls.append((kind,state,question,options))
        options=options or (['false','true'] if kind=='noul' else [str(i) for i in range(6)])
        p=[0.1/(len(options)-1)]*(len(options)-1)+[0.9]
        return {'probabilities':p,'usage':{'prompt_tokens':10}}
    def generate(self,*a,**kw):return {'text':'OK','finish_reason':'stop','usage':{'prompt_tokens':3,'completion_tokens':1}}
    def close(self):pass


def test_typesafe_batch_preserves_option_mapping_and_answers():
    r=FakeRuntime()
    with TestClient(create_app(r)) as c:
        res=c.post('/v1/systemone',json={'state':'x','questions':{
            'route':{'type':'choice','instructions':'Pick','criteria':{'billing':'Money','tech':'Code'}},
            'yes':{'type':'noul','instructions':'Is it correct?'},
            'grade':{'type':'score','instructions':'Rate','criteria':['bad','poor','fair','good','great','excellent']}}})
        assert res.status_code==200
        answers=res.json()['answers']
        assert answers['route']['choice']=='tech'
        assert answers['route']['probabilities']=={'billing':0.1,'tech':0.9}
        assert answers['yes']['noul']==0.9
        assert answers['grade']['score']==pytest.approx(4.7)
        assert r.calls[0][3]==['billing: Money','tech: Code']


def test_auth_model_and_unimplemented_controls_fail_explicitly():
    r=FakeRuntime()
    with TestClient(create_app(r,api_key='test-key')) as c:
        assert c.post('/v1/decide',json={}).status_code==401
        headers={'Authorization':'Bearer test-key'}
        assert c.post('/v1/decide',headers=headers,json={'model':'other'}).status_code==404
        assert c.post('/v1/decide',headers=headers,json={'thinking':'auto'}).status_code==400
        assert c.post('/v1/chat/completions',headers=headers,json={'stream':True}).status_code==400
        assert not r.calls


def test_jEV_score_rejects_wrong_number_of_levels():
    with pytest.raises(ValueError,match='six'):
        systemone_questions({'state':'x','questions':{'q':{'type':'score','instructions':'Rate','criteria':['bad','good']}}})
    assert format_answer('choice',[0.5,0.5],['a','b'],None)['confidence']==0
    assert format_answer('score',[1/6]*6,[str(i) for i in range(6)],list(range(6)))['confidence']==0
