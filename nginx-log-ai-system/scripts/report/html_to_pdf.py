import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = ROOT / 'logs' / 'report' / 'summary_report.html'
OUT_PDF = ROOT / 'logs' / 'report' / 'summary_report.pdf'


async def render_pdf(html_path, out_pdf):
    try:
        from pyppeteer import launch
    except Exception:
        # try to install pyppeteer
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyppeteer'])
        from pyppeteer import launch

    # Try to reuse system Chrome if available to avoid downloading Chromium
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Chromium\Application\chrome.exe",
    ]
    executable = None
    for p in chrome_paths:
        pth = Path(p)
        if pth.exists():
            executable = str(pth)
            break
    launch_args = {'args': ['--no-sandbox']}
    if executable:
        launch_args['executablePath'] = executable
    browser = await launch(**launch_args)
    page = await browser.newPage()
    await page.goto('file://' + str(html_path.resolve()), {'waitUntil': 'networkidle0'})
    await page.pdf({'path': str(out_pdf), 'format': 'A4', 'printBackground': True})
    await browser.close()


def main():
    if not HTML_PATH.exists():
        print('找不到 HTML 文件:', HTML_PATH)
        return
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print('开始使用 pyppeteer 渲染 PDF（可能会下载 Chromium，耗时）...')
    asyncio.get_event_loop().run_until_complete(render_pdf(HTML_PATH, OUT_PDF))
    print('已生成 PDF:', OUT_PDF)


if __name__ == '__main__':
    main()
