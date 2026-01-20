import json
from pathlib import Path
import glob
import os

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / 'logs' / 'evaluation'
TRAIN_DIR = ROOT / 'logs' / 'training'


def find_latest_training_zh():
    pattern = str(TRAIN_DIR / '*_zh.json')
    files = sorted(glob.glob(pattern), reverse=True)
    return Path(files[0]) if files else None


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def render_console(metrics, train_report):
    print('--- 评估摘要 (中文) ---')
    print('准确率:', metrics.get('准确率'))
    print()
    print('--- 训练报告摘要 (中文) ---')
    print('时间戳:', train_report.get('时间戳'))
    print('训练集大小:', train_report.get('训练集大小'))
    print('验证集大小:', train_report.get('验证集大小'))
    print('测试集大小:', train_report.get('测试集大小'))
    print()


def render_html(metrics, train_report, out_path: Path):
    html = ['<html><meta charset="utf-8"><body>']
    html.append('<h2>评估摘要 (中文)</h2>')
    html.append('<table border="1" cellpadding="6">')
    html.append('<tr><th>指标</th><th>值</th></tr>')
    html.append(f'<tr><td>准确率</td><td>{metrics.get("准确率")}</td></tr>')
    html.append('</table>')

    html.append('<h2>训练报告摘要 (中文)</h2>')
    html.append('<table border="1" cellpadding="6">')
    html.append('<tr><th>字段</th><th>值</th></tr>')
    html.append(f'<tr><td>时间戳</td><td>{train_report.get("时间戳")}</td></tr>')
    html.append(f'<tr><td>训练集大小</td><td>{train_report.get("训练集大小")}</td></tr>')
    html.append(f'<tr><td>验证集大小</td><td>{train_report.get("验证集大小")}</td></tr>')
    html.append(f'<tr><td>测试集大小</td><td>{train_report.get("测试集大小")}</td></tr>')
    html.append('</table>')

    # include confusion matrix image if exists
    cm_path = EVAL_DIR / 'confusion_matrix.png'
    if cm_path.exists():
        # compute path relative to the HTML output file so the image resolves correctly
        rel = os.path.relpath(cm_path, start=out_path.parent)
        rel_posix = Path(rel).as_posix()
        html.append('<h3>混淆矩阵</h3>')
        html.append(f'<img src="{rel_posix}" alt="confusion matrix" style="max-width:600px;">')

    html.append('</body></html>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))


def main():
    metrics_path = EVAL_DIR / 'metrics_zh.json'
    if not metrics_path.exists():
        print('找不到', metrics_path)
        return
    train_path = find_latest_training_zh()
    if not train_path:
        print('找不到中文训练报告 (training_report_*_zh.json)')
        return

    metrics = load_json(metrics_path)
    train_report = load_json(train_path)

    render_console(metrics, train_report)

    out_html = ROOT / 'logs' / 'report' / 'summary_report.html'
    render_html(metrics, train_report, out_html)
    print('\n已生成 HTML 报告:', out_html)


if __name__ == '__main__':
    main()
