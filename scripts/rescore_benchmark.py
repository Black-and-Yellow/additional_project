"""Recompute extraction scores from saved predictions, without inference."""
import json
from pathlib import Path
from benchmark import score_opening

root=Path(__file__).resolve().parents[1]
report=json.loads((root/'reports/kaggle-benchmark/policy-comparison.json').read_text())
report['scoring']='Opening time is the first extracted time, compared to the expected opening time, with the saved independent NLI support check. Additional supported closing times are allowed. This is an extraction proxy, not a complete correctness judgment.'
for row in report['records']:
    row['correct_time']=score_opening(row['result']['answer'],row['expected'],row['nli_supported'])
for key,summary in report['policies'].items():
    policy,correction=key.split('/')
    rows=[r for r in report['records'] if r['policy']==policy and r['correction']==correction]
    summary['time_accuracy']=sum(r['correct_time'] for r in rows)/len(rows)
report['intervention_count']=sum('intervention' in r['result'] for r in report['records'])
(root/'reports/policy-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report['policies'],indent=2))
