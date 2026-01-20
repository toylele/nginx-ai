import re


def parse_user_agent(ua: str) -> dict:
    """Very small user-agent parser returning browser and os names.

    This is heuristic-based and intentionally lightweight to avoid external deps.
    """
    if not ua:
        return {'browser': None, 'os': None}

    browser = None
    os = None

    # browser
    if 'Chrome' in ua and 'Chromium' not in ua and 'Edg/' not in ua:
        m = re.search(r'Chrome/([\d.]+)', ua)
        browser = 'Chrome ' + m.group(1) if m else 'Chrome'
    elif 'Firefox' in ua:
        m = re.search(r'Firefox/([\d.]+)', ua)
        browser = 'Firefox ' + m.group(1) if m else 'Firefox'
    elif 'Edg/' in ua:
        m = re.search(r'Edg/([\d.]+)', ua)
        browser = 'Edge ' + m.group(1) if m else 'Edge'
    elif 'Safari' in ua and 'Chrome' not in ua:
        m = re.search(r'Version/([\d.]+)', ua)
        browser = 'Safari ' + m.group(1) if m else 'Safari'
    elif 'curl' in ua.lower():
        browser = 'curl'
    else:
        browser = ua.split('/')[0]

    # os
    if 'Windows' in ua:
        os = 'Windows'
    elif 'Mac OS X' in ua or 'Macintosh' in ua:
        os = 'macOS'
    elif 'Android' in ua:
        os = 'Android'
    elif 'iPhone' in ua or 'iPad' in ua:
        os = 'iOS'
    elif 'Linux' in ua:
        os = 'Linux'

    return {'browser': browser, 'os': os}
