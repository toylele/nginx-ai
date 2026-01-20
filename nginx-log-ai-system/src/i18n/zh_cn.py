def translate_training_report(report: dict) -> dict:
    """将训练报告字典转换为中文语义版本。"""
    zh = {
        '时间戳': report.get('timestamp'),
        '模型路径': report.get('model_path'),
        '向量器路径': report.get('vectorizer_path'),
        '训练集大小': report.get('train_size'),
        '验证集大小': report.get('val_size'),
        '测试集大小': report.get('test_size'),
        '结果': {},
    }
    results = report.get('results', {}) or {}
    zh_results = {}
    if 'val_acc' in results:
        zh_results['验证准确率'] = results.get('val_acc')
    if 'val_report' in results:
        zh_results['验证分类报告'] = results.get('val_report')
    if 'test_acc' in results:
        zh_results['测试准确率'] = results.get('test_acc')
    if 'test_report' in results:
        zh_results['测试分类报告'] = results.get('test_report')
    zh['结果'] = zh_results
    return zh


def translate_metrics(metrics: dict) -> dict:
    """将评估指标转换为中文语义版本（简单映射）。"""
    zh = {
        '准确率': metrics.get('accuracy'),
        '分类报告': metrics.get('classification_report'),
    }
    return zh


# 类别标签映射（用于图表显示）
CLASS_LABELS = {0: '正常', 1: '异常'}
