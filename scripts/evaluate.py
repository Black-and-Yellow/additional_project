"""Small disclosed integration set; not a held-out research benchmark."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.app import ROOT
from backend.pipeline import Pipeline
from backend.storage import Store

parser=argparse.ArgumentParser()
parser.add_argument('--model',action='store_true')
parser.add_argument('--dataset',type=Path,default=ROOT/'data/evaluation.json')
parser.add_argument('--output',type=Path,default=ROOT/'reports/evaluation.json')
args=parser.parse_args()
model=None
if args.model:
    from backend.models import LocalModel
    model=LocalModel()
cases=json.loads(args.dataset.read_text(encoding='utf-8'))
results=[]
with tempfile.TemporaryDirectory() as tmp:
    store=Store(Path(tmp)/'eval.sqlite3',json.loads((ROOT/'data/corpus.json').read_text()))
    pipeline=Pipeline(store,model)
    for case in cases:
        r=pipeline.run(case['query'],mode='model' if model else 'lookup')
        ok=(case['expected'].lower() in r['answer'].lower()) if case['answerable'] else r['route']=='abstain'
        results.append({'case':case,'passed':ok,'result':r})
        print(case['query'],r['route'],ok,flush=True)
report={'scope':'Curated integration set. Answerable questions overlap the starter corpus. Not a generalization or hallucination benchmark.',
        'mode':'model' if model else 'lookup','cases':len(results),'passed':sum(r['passed'] for r in results),
        'mean_latency_ms':sum(r['result']['latency_ms'] for r in results)/len(results),'results':results}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
