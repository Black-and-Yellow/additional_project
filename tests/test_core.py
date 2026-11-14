import math
from contextlib import contextmanager
import pytest
from fastapi.testclient import TestClient
from backend.app import create_app
from backend.entropy import cluster_answers
from backend.retrieval import BM25, chunks


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path / 'test.sqlite3'))


def test_entropy_distribution():
    r = cluster_answers(['Sydney', 'Canberra', 'Sydney', 'Canberra', 'Sydney'])
    assert r['entropy_nats'] == pytest.approx(-.6 * math.log(.6) - .4 * math.log(.4))
    assert cluster_answers(['Gold!', 'gold'])['entropy_nats'] == 0
    assert cluster_answers(['a','b','c'])['entropy_nats'] == pytest.approx(math.log(3))


def test_complete_link_prevents_transitive_bridge():
    equivalent = lambda a,b: abs(int(a)-int(b)) <= 1
    r = cluster_answers(['1','2','3'], equivalent)
    assert [c['members'] for c in r['clusters']] == [[0,1],[2]]


def test_chunk_overlap_and_empty_index():
    result=list(chunks(' '.join(str(i) for i in range(130))))
    assert result[0].split()[-20:] == result[1].split()[:20]
    assert BM25([]).search('anything') == []


@pytest.mark.parametrize('query,expected',[
    ('Who wrote the novel Frankenstein?', 'Mary Shelley'),
    ('What is the capital of Australia?', 'Canberra'),
    ('When did the first manned moon landing occur?', 'July 20, 1969'),
    ('What element has the atomic number 79?', 'Gold'),
    ('Who painted the ceiling of the Sistine Chapel?', 'Michelangelo'),
])
def test_lookup_answers(client,query,expected):
    r=client.post('/api/run',json={'query':query}).json()
    assert expected in r['answer']
    assert r['route']=='extractive'
    assert r['uncertainty'] is None and r['diagnosis'] is None
    assert r['documents'][0]['source'].startswith('https://')
    assert client.get('/api/history').json()[0]['id']==r['id']


@pytest.mark.parametrize('query',['What is the capital of Atlantis?','What is the population of Australia?','Tell me about zebrafish genomes'])
def test_missing_evidence_abstains(client,query):
    r=client.post('/api/run',json={'query':query}).json()
    assert r['route']=='abstain'
    assert not r['documents']


def test_document_ingestion_persists(tmp_path):
    path=tmp_path/'persist.sqlite3'
    client=TestClient(create_app(path))
    r=client.post('/api/documents',json={'title':'Orion lab hours','text':'The Orion laboratory opens at 09:30 every weekday.','source':'Department guide'})
    assert r.status_code==201
    client=TestClient(create_app(path))
    r=client.post('/api/run',json={'query':'When does the Orion laboratory open?'}).json()
    assert '09:30' in r['answer']


@pytest.mark.parametrize('payload',[{'query':' '},{'query':'abc','samples':1},{'query':'abc','threshold':-1},{'query':'abc','mode':'fake'},{'query':'x'*501}])
def test_validation(client,payload):
    assert client.post('/api/run',json=payload).status_code==422


def test_missing_model_is_explicit(client):
    assert client.post('/api/run',json={'query':'What is gold?','mode':'model'}).status_code==503
    assert not client.get('/api/health').json()['model_ready']


def test_cross_origin_and_source_protocol(client):
    assert client.post('/api/run',json={'query':'test question'},headers={'origin':'https://unrelated.example'}).status_code==403
    assert client.post('/api/documents',json={'title':'Test','text':'Enough text for this document.','source':'javascript:alert(1)'}).status_code==422
    assert client.get('/../backend/app.py').status_code==404


class FakeModel:
    """Deterministic branch fixture, never used by the application."""
    model_id='test-fixture'
    calls=0
    generated_tokens=0
    varied=False
    support=True
    high_pks=False
    active=False
    def generate(self,query,docs=None,count=1,sample=False,seed=42):
        self.calls+=1
        self.generated_tokens+=count
        if self.active:
            return ['Revised evidence answer']*count
        if sample and self.varied:
            return ['Sydney','Canberra','Sydney','Canberra','Sydney'][:count]
        return ['Canberra']*count
    def equivalent(self,a,b):return a==b
    def diagnose(self,query,docs,answer):
        assert docs
        return {'ecs':.1,'pks_nats':.1 if self.high_pks else .01}
    def supports(self,answer,docs):return self.support and (not self.high_pks or answer=='Revised evidence answer')
    @contextmanager
    def intervention(self):
        self.active=True
        try:yield
        finally:self.active=False


@pytest.mark.parametrize('varied,high_pks,support,policy,route',[
    (False,False,True,'adaptive','accept_unverified'),
    (True,False,True,'adaptive','retrieval'),
    (True,True,True,'adaptive','aarf_experimental'),
    (True,True,False,'adaptive','abstain'),
    (False,False,True,'always_retrieve','retrieval'),
])
def test_model_routes(tmp_path,varied,high_pks,support,policy,route):
    model=FakeModel();model.varied=varied;model.high_pks=high_pks;model.support=support
    client=TestClient(create_app(tmp_path/'test.sqlite3',model))
    r=client.post('/api/run',json={'query':'What is the capital of Australia?','mode':'model','policy':policy})
    assert r.status_code==200,r.text
    result=r.json()
    assert result['route']==route
    assert not model.active
    if result['diagnosis']:
        names=[s['stage'] for s in result['trace']]
        assert names.index('retrieval') < names.index('diagnosis')


def test_supported_answer_does_not_get_intervened(tmp_path):
    model=FakeModel();model.varied=True;model.high_pks=True
    model.supports=lambda answer,docs: True
    client=TestClient(create_app(tmp_path/'safe.sqlite3',model))
    result=client.post('/api/run',json={'query':'What is the capital of Australia?','mode':'model'}).json()
    assert result['answer']=='Canberra'
    assert result['route']=='retrieval' and 'intervention' not in result


def test_default_model_route_requires_evidence(tmp_path):
    model=FakeModel()
    client=TestClient(create_app(tmp_path/'default.sqlite3',model))
    result=client.post('/api/run',json={'query':'What is the capital of Australia?','mode':'model'}).json()
    assert result['policy']=='always_retrieve' and result['route']=='retrieval'
    assert result['generation_calls']==1 and result['supported']


def test_no_retrieval_and_no_intervention_baselines(tmp_path):
    model=FakeModel()
    client=TestClient(create_app(tmp_path/'baseline.sqlite3',model))
    result=client.post('/api/run',json={'query':'What is the capital of Australia?','mode':'model','policy':'no_retrieval'}).json()
    assert result['generation_calls']==1 and result['uncertainty'] is None and result['documents']==[]
    result=client.post('/api/run',json={'query':'What is the capital of Australia?','mode':'model','policy':'always_retrieve','correction':'off'}).json()
    assert result['generation_calls']==1 and result['initial_answer'] is None and result['uncertainty'] is None and result['diagnosis'] is None


def test_all_claims_need_support():
    from backend.models import LocalModel
    model=object.__new__(LocalModel)
    model.entails=lambda context,claim:'Canberra' in claim
    assert model.supports('Canberra is the capital.',[{'text':'source'}])
    assert not model.supports('Canberra is the capital. Gold is blue.',[{'text':'source'}])


def test_calibration_math():
    from backend.calibration import fit,predict,metrics,select_threshold
    rows=[[.9,.01],[.8,.02],[.1,.5],[.2,.4]];labels=[0,0,1,1]
    profile=fit(rows,labels)
    scores=[predict(profile,dict(ecs=x,pks_nats=y)) for x,y in rows]
    threshold=select_threshold(labels,scores)
    assert metrics(labels,scores,threshold)['f1']==1
    assert metrics(labels,scores)['auroc']==1
    assert metrics([0,1],[.5,.5])['auroc']==.5


def test_calibration_splits_reject_source_leakage():
    from backend.calibration import validate_cases
    rows=[dict(group=split,split=split,context=split,query='question',answer='answer',hallucinated=y)
          for split in ['train','validation','test'] for y in [0,1]]
    assert len(validate_cases(rows)['test'])==1
    rows[-1]['context']='train'
    with pytest.raises(ValueError,match='overlap'):
        validate_cases(rows)


def test_opening_time_scoring_allows_supported_closing_time():
    from scripts.benchmark import score_opening
    assert score_opening('It is open at 8:00 and closes at 19:00.',['08:00'],True)
    assert not score_opening('It opens at 09:00 and closes at 08:00.',['08:00'],True)
    assert not score_opening('It opens at 08:00.',['08:00'],False)

