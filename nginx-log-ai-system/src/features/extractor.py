import sys
from pathlib import Path
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
RAW_PARSED_DIR = ROOT / 'data' / 'processed' / 'parsed_logs'
OUT_DIR = ROOT / 'data' / 'processed' / 'features'


def parse_request(request: str):
    if not request:
        return None, None, None
    parts = request.split()
    if len(parts) < 2:
        return None, None, None
    method = parts[0]
    url = parts[1]
    http_ver = parts[2] if len(parts) > 2 else None
    # get path and extension
    path = url.split('?')[0]
    ext = None
    if '.' in Path(path).name:
        ext = Path(path).suffix.lower().lstrip('.')
    return method, path, ext


def extract_features_from_record(rec: dict):
    out = {}
    out['remote_addr'] = rec.get('remote_addr')
    out['status'] = rec.get('status')
    out['body_bytes_sent'] = rec.get('body_bytes_sent')
    out['http_referer'] = rec.get('http_referer')
    out['http_user_agent'] = rec.get('http_user_agent')

    # timestamp -> hour, day_of_week
    ts = rec.get('timestamp') or rec.get('time_local')
    out['timestamp'] = ts
    try:
        if ts:
            if isinstance(ts, str):
                dt = datetime.fromisoformat(ts) if 'T' in ts else datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z")
            else:
                dt = None
            if dt:
                out['hour'] = dt.hour
                out['day_of_week'] = dt.weekday()
            else:
                out['hour'] = None
                out['day_of_week'] = None
        else:
            out['hour'] = None
            out['day_of_week'] = None
    except Exception:
        out['hour'] = None
        out['day_of_week'] = None

    # request parsing
    method, path, ext = parse_request(rec.get('request'))
    out['method'] = method
    out['path'] = path
    out['extension'] = ext
    out['request_length'] = len(rec.get('request') or '')

    # geo / ua fields if present
    out['geo_country'] = rec.get('geo_country')
    out['geo_city'] = rec.get('geo_city')
    out['geo_latitude'] = rec.get('geo_latitude')
    out['geo_longitude'] = rec.get('geo_longitude')
    out['browser'] = rec.get('browser')
    out['os'] = rec.get('os')

    return out


def extract_from_file(path: Path):
    features = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            feat = extract_features_from_record(rec)
            features.append(feat)
    return features


def save_features(features, out_path: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd:
        df = pd.DataFrame(features)
        # try parquet first
        try:
            df.to_parquet(str(out_path.with_suffix('.parquet')))
            return str(out_path.with_suffix('.parquet'))
        except Exception:
            # fallback to csv
            df.to_csv(str(out_path.with_suffix('.csv')), index=False)
            return str(out_path.with_suffix('.csv'))
    else:
        # write jsonl
        outp = out_path.with_suffix('.jsonl')
        with open(outp, 'w', encoding='utf-8') as f:
            for r in features:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        return str(outp)


def run(max_files: int = None):
    files = sorted([p for p in RAW_PARSED_DIR.iterdir() if p.is_file()])
    if max_files:
        files = files[:max_files]
    total = 0
    outputs = []
    for p in files:
        feats = extract_from_file(p)
        out_file = OUT_DIR / f'features_{p.stem}'
        saved = save_features(feats, out_file)
        print(f'Processed {p.name}: {len(feats)} rows -> {saved}')
        total += len(feats)
        outputs.append((p.name, len(feats), saved))
    print('Total features extracted:', total)
    return outputs


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--max-files', type=int, default=None)
    args = ap.parse_args()
    run(max_files=args.max_files)
