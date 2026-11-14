"""Real-weight integration check. Run after scripts/download_models.py."""
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.models import LocalModel, ROOT

model=LocalModel()
doc={'id':'canberra','text':'Canberra is the capital of Australia.'}
question='What is the capital of Australia?'
baseline=model.generate(question,[doc])[0]
diagnosis=model.diagnose(question,[doc],baseline)
assert math.isfinite(diagnosis['ecs']) and -1.001 <= diagnosis['ecs'] <= 1.001
assert 0 <= diagnosis['pks_nats'] <= math.log(2)+1e-5
assert model.equivalent('Canberra is the capital of Australia.','Australia has Canberra as its capital.')
assert not model.equivalent('Sydney is the capital of Australia.','Canberra is the capital of Australia.')
assert model.supports('Canberra is the capital of Australia.',[doc])
assert not model.supports('Sydney is the capital of Australia.',[doc])
layer=model.model.model.layers[model.layers[0]]
head_count=len(layer.self_attn.o_proj._forward_pre_hooks)
ffn_count=len(layer.mlp._forward_hooks)
with model.intervention():
    assert len(layer.self_attn.o_proj._forward_pre_hooks)==head_count+1
    assert len(layer.mlp._forward_hooks)==ffn_count+1
    revised=model.generate(question,[doc])[0]
assert len(layer.self_attn.o_proj._forward_pre_hooks)==head_count
assert len(layer.mlp._forward_hooks)==ffn_count
restored=model.generate(question,[doc])[0]
assert restored==baseline
report={'model':model.model_id,'baseline':baseline,'intervention_answer':revised,'restored_answer':restored,
        'diagnosis':diagnosis,'checks':'Finite probes, NLI paraphrase/contradiction, support, hook installation/removal, deterministic restoration passed.',
        'scope':'Integration verification only. This does not establish intervention effectiveness.'}
(ROOT/'reports').mkdir(exist_ok=True)
(ROOT/'reports/model-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
