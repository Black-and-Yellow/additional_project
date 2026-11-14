"""Package only allowlisted project source; credentials never enter the bundle."""
import base64
import argparse
import io
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['calibration','benchmark'],default='calibration')
args=parser.parse_args()
bundle=io.BytesIO()
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ['backend','scripts']:
        for p in sorted((root/folder).glob('*.py')):
            z.writestr(p.relative_to(root).as_posix(),p.read_bytes())
    z.writestr('data/calibration_cases.json',(root/'data/calibration_cases.json').read_bytes())
    if args.phase=='benchmark':
        z.writestr('artifacts/calibration.json',(root/'artifacts/calibration.json').read_bytes())
payload=base64.b64encode(bundle.getvalue()).decode('ascii')
code='''import os, sys, subprocess, base64, io, zipfile
from pathlib import Path
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['TRUSTRAG_MODEL_CACHE']='/kaggle/temp/trustrag-models'
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.6','safetensors'])
project=Path('/kaggle/temp/trustrag')
project.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
    z.extractall(project)
subprocess.check_call([sys.executable,str(project/'scripts/calibrate.py'),'--download','--output','/kaggle/working/calibration.json','--report','/kaggle/working/calibration-evaluation.json'])
'''.replace('PAYLOAD',repr(payload))
if args.phase=='benchmark':
    code=code[:code.index("subprocess.check_call([sys.executable,str(project/")]+"subprocess.check_call([sys.executable,str(project/'scripts/benchmark.py'),'--download','--profile',str(project/'artifacts/calibration.json'),'--output','/kaggle/working/policy-comparison.json'])\n"
folder=root/'kaggle'/args.phase
folder.mkdir(parents=True,exist_ok=True)
(folder/'train.py').write_text(code,encoding='utf-8')
credentials=Path.home()/'.kaggle/credentials.json'
username=json.loads(credentials.read_text())['username']
metadata=dict(id=f'{username}/trustrag-{args.phase}-v2',title=f'TrustRAG {args.phase.title()} v2',code_file='train.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=[])
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Prepared private Kaggle calibration job. No credentials or local user documents are bundled.')
