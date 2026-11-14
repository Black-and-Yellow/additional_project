"""Create a portable source bundle without caches, credentials or user records."""
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
target=root/'dist/TrustRAG_Final_Project.zip'
target.parent.mkdir(exist_ok=True)
files=[]
for name in ['backend','webapp','scripts','data','artifacts','tests','docs','presentation/final']:
    files.extend(p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.extend(root/name for name in ['README.md','run.py','Start-TrustRAG.ps1','requirements.txt','requirements-model.txt','requirements-research.txt','.gitignore'])
files.extend(root/name for name in ['reports/policy-comparison.json','reports/kaggle-calibration/calibration-evaluation.json','reports/kaggle-benchmark/policy-comparison.json','reports/release-verification.json','reports/review-workspace.png','presentation/build_review.mjs'])
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for source in sorted(files):
        archive.write(source,source.relative_to(root).as_posix())
print(f'Created {target} ({target.stat().st_size} bytes, {len(files)} files).')
