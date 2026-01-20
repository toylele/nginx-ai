from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from pathlib import Path
import joblib
import os
import time
import json
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'data' / 'models' / 'trained_models' / 'random_forest'


class PredictRequest(BaseModel):
    features: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "remote_addr": "127.0.0.1",
                    "request_method": "GET",
                    "request_path": "/",
                }
            }
        }


app = FastAPI(title='nginx-log-ai-system API')


def check_api_key(x_api_key: Optional[str] = Header(None)):
    # simple API key auth; if API_KEY not set, allow anonymous access
    expected = os.environ.get('API_KEY')
    if expected:
        if x_api_key is None or x_api_key != expected:
            raise HTTPException(status_code=401, detail='Invalid or missing API key')
    return True


def load_artifacts():
    model_path = MODEL_DIR / 'model.joblib'
    vec_path = MODEL_DIR / 'vectorizer.joblib'
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError('Model or vectorizer not found')
    clf = joblib.load(str(model_path))
    vec = joblib.load(str(vec_path))
    # attempt to read model metadata from registry if available
    registry_path = ROOT / 'data' / 'models' / 'registry.json'
    model_info = None
    try:
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as rf:
                reg = json.load(rf)
                if reg:
                    model_info = reg[-1]
    except Exception:
        model_info = None
    return clf, vec


@app.on_event('startup')
def startup_event():
    try:
        app.state.clf, app.state.vec = load_artifacts()
    except Exception as e:
        # keep app running but endpoints will report error
        app.state.clf, app.state.vec = None, None
        app.state.load_error = str(e)


@app.get('/health')
def health():
    if getattr(app.state, 'clf', None) is None:
        return {'status': 'unhealthy', 'error': getattr(app.state, 'load_error', 'model not loaded')}
    return {'status': 'ok'}


@app.post('/predict')
def predict(req: PredictRequest, authorized: bool = Depends(check_api_key)):
    if getattr(app.state, 'clf', None) is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    try:
        x = req.features
        X = app.state.vec.transform([x])
        pred = app.state.clf.predict(X)
        prob = None
        try:
            prob = float(app.state.clf.predict_proba(X)[:, 1][0])
        except Exception:
            prob = None

        out = {'prediction': int(pred[0]), 'probability': prob}

        # log prediction to logs/predictions/predictions.jsonl
        try:
            log_dir = ROOT / 'logs' / 'predictions'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / 'predictions.jsonl'
            entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'features': x,
                'prediction': int(pred[0]),
                'probability': prob,
            }
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception:
            pass

        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
