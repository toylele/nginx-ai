"""
国际化模块 - 中文本地化

本模块提供将系统输出转换为中文语义的功能。
包括：
1. 训练报告翻译 - 将英文键转换为中文
2. 评估指标翻译 - 将性能指标名称中文化
3. 类别标签映射 - 提供预测结果的中文表示

用途：
- 报表生成 - 在HTML/PDF报告中使用中文字段名
- 混淆矩阵标签 - 在图表中显示中文类别标签
- 用户界面 - 为用户提供本地化的文字提示

作者: nginx-log-ai-system Team
"""


def translate_training_report(report: dict) -> dict:
    """
    将训练报告字典转换为中文语义版本
    
    将英文的训练报告字段名翻译为中文，便于非英文用户理解。
    
    参数:
        report: 原始英文训练报告字典，包含以下字段：
            - timestamp: 训练时间
            - model_path: 模型保存路径
            - vectorizer_path: 向量化器保存路径
            - train_size: 训练集大小
            - val_size: 验证集大小
            - test_size: 测试集大小
            - results: 包含性能指标的字典
    
    返回:
        dict: 中文化的训练报告，包含以下字段：
            - 时间戳: 训练时间
            - 模型路径: 模型文件位置
            - 向量器路径: 向量化器文件位置
            - 训练集大小: 用于训练的样本数
            - 验证集大小: 用于验证的样本数
            - 测试集大小: 用于测试的样本数
            - 结果: 包含中文化指标的字典
                - 验证准确率
                - 验证分类报告
                - 测试准确率
                - 测试分类报告
    
    示例:
        >>> report = {
        ...     'timestamp': '2026-01-20 10:00:00',
        ...     'train_size': 5000,
        ...     'results': {'test_acc': 0.95}
        ... }
        >>> zh_report = translate_training_report(report)
        >>> print(zh_report['准确率'])
    """
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
    
    # 翻译性能指标
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
    """
    将评估指标转换为中文语义版本
    
    提供模型评估指标的中文翻译。用于生成易于理解的
    性能评估报告。
    
    参数:
        metrics: 原始英文指标字典，包含：
            - accuracy: 模型整体准确率
            - classification_report: 详细的分类性能指标
    
    返回:
        dict: 中文化的指标字典
            - 准确率: 模型的总体准确率
            - 分类报告: 按类别的详细指标
                - precision (精准率)
                - recall (召回率)
                - f1-score (F1分数)
                - support (样本数)
    
    示例:
        >>> metrics = {'accuracy': 0.92, 'classification_report': {...}}
        >>> zh_metrics = translate_metrics(metrics)
        >>> print(zh_metrics['准确率'])  # 输出: 0.92
    """
    zh = {
        '准确率': metrics.get('accuracy'),
        '分类报告': metrics.get('classification_report'),
    }
    return zh


# 预测结果类别标签映射（用于图表和报告显示）
# 这个字典将数值预测结果映射到中文标签
CLASS_LABELS = {
    0: '正常',  # 状态码 < 400 的请求
    1: '异常'   # 状态码 >= 400 的请求
}
