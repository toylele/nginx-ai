import sys
from pathlib import Path
import argparse

# ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collector import read_lines
from src.data.cleaner.data_cleaner import clean_line
from src.data.cleaner.normalizer import normalize_timestamp
from src.data.cleaner.enricher import GeoIPEnricher
from src.features.utils.user_agent import parse_user_agent
from src.data.storage.file_storage import save_json_lines
from src.data.storage import database


RAW_DIR = ROOT / 'data' / 'raw' / 'nginx_logs'
OUT_DIR = ROOT / 'data' / 'processed' / 'parsed_logs'
DB_PATH = ROOT / 'data' / 'models' / 'logs.db'


def process_file(path: Path, max_lines: int = None, to_db: bool = False):
    records = []
    enricher = GeoIPEnricher(ROOT)
    for line in read_lines(str(path), max_lines=max_lines):
        rec = clean_line(line)
        if not rec:
            continue
        rec = normalize_timestamp(rec)
        # enrich geo data
        rec = enricher.enrich(rec)
        # parse user agent
        ua = rec.get('http_user_agent')
        ua_info = parse_user_agent(ua)
        rec.update(ua_info)
        records.append(rec)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'parsed_{path.name}.jsonl'
    save_json_lines(records, str(out_path))

    if to_db:
        conn = database.init_db(str(DB_PATH))
        for r in records:
            database.insert_log(conn, r)
        conn.close()

    return len(records), out_path


def process_all(max_lines: int = None, to_db: bool = False):
    files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    total = 0
    outputs = []
    for f in files:
        n, out_path = process_file(f, max_lines=max_lines, to_db=to_db)
        print(f'Processed {f.name}: {n} records -> {out_path}')
        total += n
        outputs.append((f.name, n, out_path))
    print('Total records processed:', total)
    return total, outputs


def main():
    parser = argparse.ArgumentParser(description='Data pipeline for nginx logs')
    parser.add_argument('--max-lines', type=int, default=None, help='Max lines to process per file')
    parser.add_argument('--to-db', action='store_true', dest='to_db', help='Persist parsed records into sqlite DB')
    args = parser.parse_args()

    process_all(max_lines=args.max_lines, to_db=args.to_db)


if __name__ == '__main__':
    main()
