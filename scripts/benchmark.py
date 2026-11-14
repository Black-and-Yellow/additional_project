"""Compare executable policies on untouched controlled test groups."""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.models import LocalModel,ROOT
from backend.pipeline import Pipeline
from backend.storage import Store


def times(text):
    return [f'{int(hour):02d}:{minute}' for hour,minute in re.findall(r'\b(\d{1,2}):(\d{2})\b',text)]


def score_opening(answer, expected, supported):
    actual=times(answer)
    return bool(actual and expected and supported and actual[0]==expected[0])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,default=ROOT/'artifacts/calibration.json')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/policy-comparison.json')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    cases=json.loads((ROOT/'data/calibration_cases.json').read_text())
    cases=[c for c in cases if c['split']=='test' and c['hallucinated']==0]
    model=LocalModel(local_only=not args.download,calibration_path=args.profile)
    documents=[{'id':c['group'],'title':c['group']+' laboratory hours','text':c['context'],'source':'Controlled fictional handbook'} for c in cases]
    policies=[('no_retrieval','off'),('always_retrieve','off'),('adaptive','off'),('adaptive','guarded')]
    report={'scope':'Controlled fictional handbooks, test source groups excluded from training and validation. Time extraction scores the stated opening time; no public benchmark claims.',
        'device':model.device,'cases':len(cases),'policies':{},'records':[], 'status':'running',
        'implementation_version':'2026-11-15',
        'methodology':'Always-retrieve goes directly to contextual generation. Adaptive and no-retrieval include a context-free answer. Actual elapsed time and generated tokens are recorded.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline=Pipeline(Store(Path(tmp)/'benchmark.sqlite3',documents),model)
        for policy,correction in policies:
            records=[]
            for case in cases:
                r=pipeline.run(case['query'],mode='model',policy=policy,correction=correction,samples=3)
                expected=times(case['answer'])
                actual=times(r['answer'])
                faithful=r['route']!='abstain' and model.supports(r['answer'],[{'text':case['context']}])
                # All benchmark questions ask the opening time. An additional
                # supported closing time must not turn a correct answer into a failure.
                record={'group':case['group'],'expected':expected,'correct_time':score_opening(r['answer'],expected,faithful),'nli_supported':faithful,'result':r}
                records.append(record)
                report['records'].append({'policy':policy,'correction':correction,**record})
                args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(policy,correction,case['group'],record['correct_time'],r['route'],flush=True)
            report['policies'][policy+'/'+correction]={'time_accuracy':sum(x['correct_time'] for x in records)/len(records),
                'nli_support_rate':sum(x['nli_supported'] for x in records)/len(records),
                'abstention_rate':sum(x['result']['route']=='abstain' for x in records)/len(records),
                'mean_latency_ms':sum(x['result']['latency_ms'] for x in records)/len(records),
                'generated_tokens':sum(x['result']['generated_tokens'] for x in records)}
    report['status']='complete'
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['policies'],indent=2))

if __name__=='__main__':main()
