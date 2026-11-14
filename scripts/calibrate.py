"""Train selection on train groups, set threshold on validation, test once."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.models import LocalModel, ROOT, REVISIONS
from backend.calibration import fit,predict,metrics,select_threshold,validate_cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/calibration_cases.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/calibration.json')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/calibration-evaluation.json')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    cases=json.loads(args.dataset.read_text(encoding='utf-8'))
    groups=validate_cases(cases)
    model=LocalModel(local_only=not args.download)
    n=len(model.model.model.layers)
    model.layers=sorted(set([n//2,n-4,n-2,n-1]))
    model.heads=list(range(min(4,model.model.config.num_attention_heads)))
    measured=[]
    for i,case in enumerate(cases):
        d=model.diagnose(case['query'],[{'id':case['group'],'text':case['context']}],case['answer'])
        measured.append({'case':case,'diagnosis':d})
        print(f"Measured {i+1}/{len(cases)} ({case['split']})",flush=True)
    train=[r for r in measured if r['case']['split']=='train']
    val=[r for r in measured if r['case']['split']=='validation']
    test=[r for r in measured if r['case']['split']=='test']
    def features(rows,key):
        return [[r['diagnosis']['components']['ecs'][key],r['diagnosis']['components']['pks'][key.split(':')[0]]] for r in rows]
    def score(profile,rows,key):
        return [predict(profile,{'ecs':x[0],'pks_nats':x[1]}) for x in features(rows,key)]
    ytrain=[r['case']['hallucinated'] for r in train]
    candidates=[]
    for key in train[0]['diagnosis']['components']['ecs']:
        profile=fit(features(train,key),ytrain)
        candidates.append((metrics(ytrain,score(profile,train,key))['auroc'],key,profile))
    _,key,profile=max(candidates,key=lambda x:(x[0],x[1]))
    yval=[r['case']['hallucinated'] for r in val]
    threshold=select_threshold(yval,score(profile,val,key))
    profile.update(schema_version=1,model_id=model.model_id,revision=REVISIONS.get(model.model_id),
        layers=[int(key.split(':')[0])],heads=[int(key.split(':')[1])],threshold=threshold,
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        scope='Controlled context-consistency calibration. Not a public benchmark or a faithful ReDeEP reproduction.',
        split_groups={s:sorted(v) for s,v in groups.items()})
    report={'profile':profile,'metrics':{s:metrics([r['case']['hallucinated'] for r in rows],score(profile,rows,key),threshold) for s,rows in [('train',train),('validation',val),('test',test)]},
        'baselines':{name:metrics([r['case']['hallucinated'] for r in test],values,.5) for name,values in [('always_faithful',[0.]*len(test)),('always_hallucinated',[1.]*len(test))]},
        'measurements':measured,'device':model.device}
    for path,obj in [(args.output,profile),(args.report,report)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps(report['metrics'],indent=2),flush=True)

if __name__=='__main__':main()
