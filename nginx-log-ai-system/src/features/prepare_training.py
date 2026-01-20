import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / 'data' / 'processed' / 'features'
TRAIN_DIR = ROOT / 'data' / 'processed' / 'training' / 'train'
CFG_PATH = ROOT / 'configs' / 'feature_config.yaml'


def load_feature_config(cfg_path: Path):
    if not cfg_path.exists():
        return None
    try:
        import yaml
    except Exception:
        yaml = None
    if yaml:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    # fallback simple parser
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
                cur[k] = v or None
    return {'features': features}


def find_processed_files(feature_dir: Path):
    files = []
    for p in feature_dir.iterdir():
        if p.is_file() and p.name.startswith('processed_'):
            files.append(p)
    # also accept processed_features_*.parquet/csv/jsonl
    if not files:
        for p in feature_dir.iterdir():
            if p.is_file() and 'processed' in p.name:
                files.append(p)
    return sorted(files)


def read_records(path: Path):
    # try parquet/csv with pandas if available
    try:
        import pandas as pd
    except Exception:
        pd = None
    if pd and path.suffix == '.parquet':
        df = pd.read_parquet(str(path))
        return df.to_dict(orient='records')
    if pd and path.suffix == '.csv':
        df = pd.read_csv(str(path))
        return df.to_dict(orient='records')
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
    return rows


def label_record(rec: dict) -> int:
    # simple rule-based label: status >= 400 => anomaly (1), else normal (0)
    s = rec.get('status')
    try:
        s = int(s)
    except Exception:
        return 0
    return 1 if s >= 400 else 0


def select_feature_columns(rec: dict, cfg: dict):
    # choose features marked as role numeric or categorical in config
    if not cfg:
        # default selection
        keys = ['status','body_bytes_sent','hour','request_length','geo_latitude','geo_longitude','method','extension','geo_country','browser','os']
        return [k for k in keys if k in rec]
    feat_defs = cfg.get('features', [])
    cols = []
    for f in feat_defs:
        name = f.get('name')
        if name in rec and f.get('role') in ('numeric','categorical'):
            cols.append(name)
    return cols


def run():
    files = find_processed_files(FEATURE_DIR)
    if not files:
        print('No processed feature files found in', FEATURE_DIR)
        return

    cfg = load_feature_config(CFG_PATH)
    all_X = []
    all_y = []
    total = 0
    for p in files:
        recs = read_records(p)
        for r in recs:
            cols = select_feature_columns(r, cfg)
            x = {c: r.get(c) for c in cols}
            y = label_record(r)
            all_X.append(x)
            all_y.append({'label': y})
        print(f'Loaded {len(recs)} records from {p.name}')
        total += len(recs)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    x_path = TRAIN_DIR / 'X_train.jsonl'
    y_path = TRAIN_DIR / 'y_train.jsonl'
    with open(x_path, 'w', encoding='utf-8') as fx, open(y_path, 'w', encoding='utf-8') as fy:
        for xi, yi in zip(all_X, all_y):
            fx.write(json.dumps(xi, ensure_ascii=False) + '\n')
            fy.write(json.dumps(yi, ensure_ascii=False) + '\n')

    print('Wrote X_train ->', x_path)
    print('Wrote y_train ->', y_path)
    print('Total rows:', total)


if __name__ == '__main__':
    run()
