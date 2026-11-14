"""Fit a reproducible two-signal detector without changing LLM weights."""
import math
import numpy as np


def validate_profile(profile):
    if profile.get('schema_version') != 1:
        raise ValueError('Unsupported calibration schema.')
    for field, size in [('mean',2),('scale',2),('weights',3)]:
        values=profile.get(field,[])
        if len(values)!=size or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):
            raise ValueError(f'Invalid calibration {field}.')
    if any(x<=0 for x in profile['scale']):
        raise ValueError('Calibration scales must be positive.')
    threshold=profile.get('threshold')
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Calibration threshold must be between zero and one.')
    for name in ['layers','heads']:
        if not profile.get(name) or any(type(x) is not int or x<0 for x in profile[name]):
            raise ValueError('Calibration component indexes must be nonnegative integers.')


def validate_cases(cases):
    groups={s:set() for s in ['train','validation','test']}
    contexts={s:set() for s in groups}
    for row in cases:
        if row['split'] not in groups or row['hallucinated'] not in (0,1):
            raise ValueError('Invalid split or hallucination label.')
        if not all(isinstance(row[k],str) and row[k].strip() for k in ['group','context','query','answer']):
            raise ValueError('All calibration text fields must be non-empty.')
        groups[row['split']].add(row['group'])
        contexts[row['split']].add(' '.join(row['context'].lower().split()))
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if groups[a]&groups[b] or contexts[a]&contexts[b]:
            raise ValueError('Source groups or identical contexts overlap across splits.')
    for split in groups:
        if {r['hallucinated'] for r in cases if r['split']==split}!={0,1}:
            raise ValueError('Every split requires both label classes.')
    return groups


def auc(labels, scores):
    positive = [s for y,s in zip(labels,scores) if y]
    negative = [s for y,s in zip(labels,scores) if not y]
    if not positive or not negative:
        return None
    return sum(float(a>b)+0.5*float(a==b) for a in positive for b in negative)/(len(positive)*len(negative))


def metrics(labels, scores, threshold=0.5):
    tp=fp=tn=fn=0
    for y,s in zip(labels,scores):
        pred=s>=threshold
        tp+=bool(y and pred); fp+=bool(not y and pred)
        tn+=bool(not y and not pred); fn+=bool(y and not pred)
    return dict(count=len(labels),tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),
        precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),
        accuracy=(tp+tn)/max(1,len(labels)),auroc=auc(labels,scores),
        brier=sum((s-y)**2 for y,s in zip(labels,scores))/max(1,len(labels)))


def fit(rows, labels):
    x=np.asarray(rows,dtype=float)
    mean=x.mean(0); scale=np.maximum(x.std(0),1e-6)
    z=(x-mean)/scale
    design=np.column_stack([np.ones(len(z)),z]); weights=np.zeros(3)
    for _ in range(1500):
        p=1/(1+np.exp(-np.clip(design@weights,-30,30)))
        penalty=np.r_[0,weights[1:]*0.01]
        weights-=0.05*(design.T@(p-np.asarray(labels))/len(labels)+penalty)
    return dict(mean=mean.tolist(),scale=scale.tolist(),weights=weights.tolist())


def predict(profile, diagnosis):
    values=[diagnosis['ecs'],diagnosis['pks_nats']]
    z=[(v-m)/s for v,m,s in zip(values,profile['mean'],profile['scale'])]
    logit=profile['weights'][0]+sum(a*b for a,b in zip(z,profile['weights'][1:]))
    return 1/(1+math.exp(-max(-30,min(30,logit))))


def select_threshold(labels, scores):
    candidates=sorted(set([0.,1.]+list(scores)))
    return max(candidates,key=lambda t:(metrics(labels,scores,t)['f1'],-abs(t-.5)))
