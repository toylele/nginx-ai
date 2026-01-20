import os
import sys
from pathlib import Path

# ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('API_KEY', 'testkey')

def main():
    import importlib
    import threading
    import time
    import requests
    import uvicorn

    appmod = importlib.import_module('src.api.app')
    # debug: print model dir and files as seen by the imported module
    try:
        print('DEBUG ROOT=', appmod.ROOT)
        print('DEBUG MODEL_DIR=', appmod.MODEL_DIR)
        print('DEBUG model exists', (appmod.MODEL_DIR / 'model.joblib').exists())
        print('DEBUG vec exists', (appmod.MODEL_DIR / 'vectorizer.joblib').exists())
    except Exception as e:
        print('DEBUG failed to inspect model paths:', e)

    def run_uvicorn():
        uvicorn.run(appmod.app, host='127.0.0.1', port=8001, log_level='info')

    t = threading.Thread(target=run_uvicorn, daemon=True)
    t.start()
    time.sleep(1)

    print('Calling /health...')
    try:
        r = requests.get('http://127.0.0.1:8001/health')
        print('health', r.status_code, r.json())
    except Exception as e:
        print('health request failed:', e)

    print('Calling /predict...')
    payload = {
        'features': {
            'remote_addr': '127.0.0.1',
            'request_method': 'GET',
            'request_path': '/',
        }
    }
    headers = {'x-api-key': os.environ.get('API_KEY')}
    try:
        r2 = requests.post('http://127.0.0.1:8001/predict', json=payload, headers=headers, timeout=10)
        print('predict', r2.status_code, r2.json())
    except Exception as e:
        print('predict request failed:', e)

    # give server time to write log
    time.sleep(0.5)

    # show last prediction log if exists
    log_path = ROOT / 'logs' / 'predictions' / 'predictions.jsonl'
    if log_path.exists():
        print('\nLast prediction log line:')
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
            if lines:
                print(lines[-1])
            else:
                print('(no lines)')
    else:
        print('No prediction log found at', log_path)


if __name__ == '__main__':
    main()
