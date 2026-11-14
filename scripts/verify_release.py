"""Real-weight checks for the reviewed release; cached assets only."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.models import LocalModel,ROOT
from backend.pipeline import Pipeline
from backend.storage import Store

model=LocalModel(calibration_path=ROOT/'artifacts/calibration.json')
docs=json.loads((ROOT/'data/corpus.json').read_text())
with tempfile.TemporaryDirectory() as tmp:
    pipeline=Pipeline(Store(Path(tmp)/'verify.sqlite3',docs),model)
    answer=pipeline.run('What is the capital of Australia?',mode='model',policy='always_retrieve',correction='off')
    assert answer['generation_calls']==1 and answer['initial_answer'] is None
    assert 'Canberra' in answer['answer'] and answer['supported']
    numeric_rejected=not model.supports('The laboratory opens at 09:30.',[{'text':'The laboratory opens at 09:00.'}])
    assert numeric_rejected
    profile=model.diagnose('What is the capital of Australia?',answer['documents'],answer['answer'])
    assert profile['calibrated']
    result={'query':answer,'numeric_mismatch_rejected':numeric_rejected,'diagnosis':profile,'device':model.device,
        'scope':'Release smoke checks only; November 13 policy comparison remains a historical experiment.'}
    (ROOT/'reports/release-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'answer':answer['answer'],'generation_calls':answer['generation_calls'],'numeric_mismatch_rejected':numeric_rejected,'calibrated':profile['calibrated']},indent=2))
