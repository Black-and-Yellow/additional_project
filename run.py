"""Start the local TrustRAG application."""
import argparse
import uvicorn
from backend.app import create_app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store_true', help='Load cached local LLM and NLI weights')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--calibration', help='Path to fitted calibration JSON')
    args = parser.parse_args()
    model = None
    if args.model:
        from backend.models import LocalModel
        print('Loading local model and NLI. First startup can take a minute.', flush=True)
        from pathlib import Path
        default_profile = Path(__file__).resolve().parent / 'artifacts/calibration.json'
        model = LocalModel(calibration_path=args.calibration or (default_profile if default_profile.exists() else None))
    uvicorn.run(create_app(model=model), host='127.0.0.1', port=args.port)

