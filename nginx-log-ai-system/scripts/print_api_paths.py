import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib
mod = importlib.import_module('src.api.app')
print('ROOT=', mod.ROOT)
print('MODEL_DIR=', mod.MODEL_DIR)
print('model exists', (mod.MODEL_DIR / 'model.joblib').exists())
print('vec exists', (mod.MODEL_DIR / 'vectorizer.joblib').exists())
