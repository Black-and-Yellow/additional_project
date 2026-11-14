"""Explicit one-time download; application startup never downloads silently."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
from backend.models import GENERATOR, NLI_MODEL, ROOT, REVISIONS

for model_id in (GENERATOR, NLI_MODEL):
    print(f'Downloading {model_id}', flush=True)
    path = snapshot_download(model_id, revision=REVISIONS[model_id], cache_dir=ROOT / '.models',
        allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors'],
        ignore_patterns=['onnx/*', 'openvino/*'], max_workers=2)
    print(f'Cached at {path}', flush=True)
