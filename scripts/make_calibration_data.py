"""Controlled fictional records; labels concern context fidelity only."""
import json
from pathlib import Path
names=['Aster','Birch','Cedar','Dahlia','Elm','Fir','Grove','Hazel','Iris','Juniper','Kestrel','Larch','Maple','Nectar','Olive','Pine','Quartz','Reed','Spruce','Thistle','Umber','Violet','Willow','Xenia','Yarrow','Zinnia','Amber','Beryl','Coral','Delta','Ember','Flint']
cases=[]
for i,name in enumerate(names):
    split='train' if i<16 else ('validation' if i<24 else 'test')
    hour=8+i%8; alternate=8+(i+3)%8
    context=f'In the fictional {name} laboratory handbook, the {name} laboratory opens at {hour:02d}:00. It closes at 19:00. The coordinator is Dr. {name}.'
    for wrong in [False,True]:
        cases.append(dict(group=name.lower(),split=split,query=f'When does the {name} laboratory open?',context=context,
            answer=f'The {name} laboratory opens at {alternate if wrong else hour:02d}:00.',hallucinated=int(wrong)))
path=Path(__file__).resolve().parents[1]/'data/calibration_cases.json'
path.write_text(json.dumps(cases,indent=2),encoding='utf-8')
print(f'Wrote {len(cases)} controlled cases with disjoint source groups.')
