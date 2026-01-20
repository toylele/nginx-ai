from src.data.parser.nginx_parser import parse_line


def test_parse_line():
    line = '127.0.0.1 - - [10/Oct/2000:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 123 "-" "curl/7.58.0"'
    rec = parse_line(line)
    assert rec is not None
    assert rec['remote_addr'] == '127.0.0.1'
    assert rec['status'] == '200'
