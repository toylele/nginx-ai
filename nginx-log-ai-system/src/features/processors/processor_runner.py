import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except Exception:
    pd = None

def load_feature_config(root: Path):
    cfg_path = root / 'configs' / 'feature_config.yaml'
    if not cfg_path.exists():
        return None
    try:
        import yaml
    except Exception:
        yaml = None

    if yaml:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # very small fallback YAML parser for our simple config structure
    features = []
    cur = None
    with open(cfg_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.strip().startswith('- name:'):
                name = line.split(':', 1)[1].strip()
                cur = {'name': name}
                features.append(cur)
            elif ':' in line and cur is not None:
                k, v = line.split(':', 1)
                k = k.strip().lstrip('- ')
                v = v.strip()
                if v == '':
                    v = None
                cur[k] = v
    return {'features': features}

from src.features.processors.encoder import fit_encoder, transform as encode_transform
from src.features.processors.scaler import fit_scaler, transform as scale_transform

RAW_FEATURE_DIR = ROOT / 'data' / 'processed' / 'features'
OUT_DIR = ROOT / 'data' / 'processed' / 'features'
META_PATH = OUT_DIR / 'feature_metadata.json'

# Load feature configuration (if present) to determine categorical/numeric columns
_feature_cfg = load_feature_config(ROOT) or {}
_features_list = _feature_cfg.get('features') or []

CATEGORICAL_COLS = [f['name'] for f in _features_list if f.get('type') == 'categorical']
NUMERIC_COLS = [f['name'] for f in _features_list if f.get('type') in ('integer', 'float')]

# sensible defaults if config not present or missing entries
if not CATEGORICAL_COLS:
    CATEGORICAL_COLS = ['method', 'extension', 'geo_country', 'geo_city', 'browser', 'os']
if not NUMERIC_COLS:
    NUMERIC_COLS = ['status', 'body_bytes_sent', 'hour', 'request_length', 'geo_latitude', 'geo_longitude']


def load_features(path: Path):
    if pd is not None and path.suffix in ['.parquet', '.csv']:
        if path.suffix == '.parquet':
            return pd.read_parquet(str(path))
        else:
            return pd.read_csv(str(path))
    # fallback jsonl
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def save_metadata(meta: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # include feature_config content if available
    if _feature_cfg:
        meta['feature_config'] = _feature_cfg
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def run(max_files: int = None):
    files = sorted([p for p in RAW_FEATURE_DIR.iterdir() if p.is_file() and p.name.startswith('features_')])
    if max_files:
        files = files[:max_files]
    meta = {'encoders': {}, 'scalers': {}}
    for p in files:
        df = load_features(p)
        print('Loaded', p, 'rows:', len(df) if hasattr(df, '__len__') else 'unknown')
        # fit encoders and scalers
        enc_map = fit_encoder(df, CATEGORICAL_COLS)
        meta['encoders'][p.name] = enc_map
        df_encoded = encode_transform(df, enc_map, CATEGORICAL_COLS)
        scaler_params = fit_scaler(df_encoded, NUMERIC_COLS)
        meta['scalers'][p.name] = scaler_params
        df_scaled = scale_transform(df_encoded, scaler_params, NUMERIC_COLS)
        # save
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f'processed_{p.stem}.parquet'
        if pd is not None:
            try:
                df_scaled.to_parquet(str(out_path))
                print('Saved processed features to', out_path)
            except Exception:
                out_csv = out_path.with_suffix('.csv')
                df_scaled.to_csv(str(out_csv), index=False)
                print('Saved processed features to', out_csv)
        else:
            out_jsonl = out_path.with_suffix('.jsonl')
            with open(out_jsonl, 'w', encoding='utf-8') as f:
                for r in df_scaled:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print('Saved processed features to', out_jsonl)
    save_metadata(meta)
    print('Saved feature metadata to', META_PATH)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-files', type=int, default=None)
    args = ap.parse_args()
    run(max_files=args.max_files)
