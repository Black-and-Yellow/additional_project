import json
import threading
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Literal
from .entropy import cluster_answers
from .pipeline import Pipeline
from .storage import Store

ROOT = Path(__file__).resolve().parents[1]


class Query(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    query: str = Field(min_length=3, max_length=500)
    mode: Literal['lookup', 'model'] = 'lookup'
    samples: int = Field(default=5, ge=2, le=8)
    threshold: float = Field(default=0.65, ge=0, le=2.1, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2147483647)
    policy: Literal['adaptive', 'always_retrieve', 'no_retrieval'] = 'always_retrieve'
    correction: Literal['guarded', 'off'] = 'guarded'


class Document(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    title: str = Field(min_length=3, max_length=150)
    text: str = Field(min_length=20, max_length=20000)
    source: str = Field(default='User-supplied document', max_length=500)

    @field_validator('source')
    @classmethod
    def validate_source(cls, value):
        if ':' in value and urlparse(value).scheme not in ('http', 'https'):
            raise ValueError('Use an http(s) URL or a plain source label.')
        return value


class EntropyRequest(BaseModel):
    samples: list[str] = Field(min_length=2, max_length=8)

    @field_validator('samples')
    @classmethod
    def valid_samples(cls, values):
        if any(not v.strip() or len(v) > 1000 for v in values):
            raise ValueError('Each sample must contain 1–1000 characters.')
        return values


def create_app(db_path=None, model=None):
    app = FastAPI(title='TrustRAG', version='1.1.0')
    store = Store(db_path or ROOT / '.data/trustrag.sqlite3', json.loads((ROOT / 'data/corpus.json').read_text(encoding='utf-8')))
    pipeline = Pipeline(store, model)
    lock = threading.Lock()

    @app.middleware('http')
    async def local_origin(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method == 'POST' and origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Cross-origin writes are not allowed.'}, status_code=403)
        return await call_next(request)

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'project': 'TrustRAG', 'model_ready': model is not None,
                'calibrated': bool(getattr(model, 'calibration', None)),
                'device': getattr(model, 'device', None),
                'model': model.model_id if model else None, 'documents': len(store.documents())}

    @app.post('/api/run')
    def run(payload: Query):
        if not lock.acquire(blocking=False):
            raise HTTPException(409, 'An experiment is already running. Please wait.')
        try:
            return pipeline.run(**payload.model_dump())
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        finally:
            lock.release()

    @app.get('/api/documents')
    def documents():
        return store.documents()

    @app.post('/api/documents', status_code=201)
    def add_document(payload: Document):
        return store.add_document(**payload.model_dump())

    @app.get('/api/history')
    def history():
        return store.history()

    @app.get('/api/research')
    def research():
        calibration_path = ROOT / 'reports/kaggle-calibration/calibration-evaluation.json'
        benchmark_path = ROOT / 'reports/policy-comparison.json'
        result = {'scope': 'Controlled fictional context-consistency examples. General factual reliability is untested.'}
        if calibration_path.exists():
            saved = json.loads(calibration_path.read_text(encoding='utf-8'))
            result['calibration'] = {'layers': saved['profile']['layers'], 'heads': saved['profile']['heads'],
                                     'test': saved['metrics']['test']}
        if benchmark_path.exists():
            saved = json.loads(benchmark_path.read_text(encoding='utf-8'))
            result['benchmark'] = {'policies': saved['policies'], 'cases': saved['cases'],
                                   'status': saved.get('status', 'complete')}
        return result

    @app.post('/api/entropy')
    def entropy(payload: EntropyRequest):
        result = cluster_answers(payload.samples)
        result['method'] = 'Normalized exact-text clustering (not semantic NLI)'
        return result

    app.mount('/', StaticFiles(directory=ROOT / 'webapp', html=True), name='webapp')
    return app

