from pathlib import Path
import sys

# ensure project root is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.utils.user_agent import parse_user_agent


def test_parse_curl_user_agent():
    ua = 'curl/7.58.0'
    info = parse_user_agent(ua)
    assert info['browser'] == 'curl'


def test_parse_chrome_user_agent():
    ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    info = parse_user_agent(ua)
    assert info['browser'] is not None and 'Chrome' in info['browser']
    assert info['os'] == 'Windows'


def test_parse_safari_user_agent():
    ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15'
    info = parse_user_agent(ua)
    assert info['browser'] is not None and 'Safari' in info['browser']
    assert info['os'] == 'macOS'
