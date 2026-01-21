"""
基线模型训练脚本 - 随机森林异常检测模型

本脚本实现了基线机器学习模型的训练流程：
1. 数据加载 - 从 JSONL 文件读取训练特征和标签
2. 数据分割 - 将数据分为训练集、验证集、测试集
3. 特征向量化 - 使用 DictVectorizer 将特征字典转换为矩阵
4. 模型训练 - 训练随机森林分类器
5. 概率校准 - 使用 CalibratedClassifierCV 校准预测概率
6. 模型评估 - 在验证集和测试集上评估模型性能
7. 结果保存 - 保存模型、向量化器、报告和模型注册表

输出文件：
- data/models/trained_models/random_forest/model.joblib - 训练好的分类器
- data/models/trained_models/random_forest/vectorizer.joblib - 特征向量化器
- logs/training/training_report_*.json - 训练报告（英文）
- logs/training/training_report_*_zh.json - 训练报告（中文）
- data/models/registry.json - 模型注册表（记录所有训练的模型）

作者: nginx-log-ai-system Team
"""

import sys
from pathlib import Path
import json

# 项目根目录和关键数据目录配置
ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = ROOT / 'data' / 'processed' / 'training' / 'train'
MODEL_DIR = ROOT / 'data' / 'models' / 'trained_models' / 'random_forest'

# 确保项目根目录在 Python 路径中，便于导入模块
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path):
    """
    从 JSONL 文件读取所有记录
    
    JSONL (JSON Lines) 格式：每行一个有效的 JSON 对象。
    本函数能够处理格式错误的行，跳过它们以保证数据管道的健壮性。
    
    参数:
        path: JSONL 文件路径
    
    返回:
        list: 解析后的字典列表
    
    示例:
        >>> rows = read_jsonl(Path('features.jsonl'))
        >>> print(f'加载了 {len(rows)} 条记录')
    """
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                rows.append(json.loads(line))
            except Exception:
                # 跳过格式错误的行，不中断处理
                continue
    return rows


def ensure_deps():
    """
    检查并安装必要的依赖包
    
    检查是否已安装 scikit-learn、joblib 和 pandas。
    如果未安装，自动使用 pip 安装。
    
    这样可以使脚本更具可移植性，用户不需要手动管理依赖。
    """
    try:
        import sklearn  # noqa: F401
        import joblib
        import pandas as pd
    except Exception:
        # 如果依赖缺失，使用 pip 自动安装
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                             'scikit-learn', 'joblib', 'pandas'])


def run():
    """
    主训练函数
    
    执行完整的模型训练流程：
    1. 加载训练数据（特征和标签）
    2. 验证数据完整性
    3. 分割数据为训练、验证、测试集
    4. 向量化特征（将字典转换为数值矩阵）
    5. 训练随机森林模型
    6. 校准概率预测（提高概率预测的准确性）
    7. 在验证集和测试集上评估性能
    8. 保存模型和相关文件
    9. 更新模型注册表
    """
    # 定义训练和标签文件路径
    X_path = TRAIN_DIR / 'X_train.jsonl'
    y_path = TRAIN_DIR / 'y_train.jsonl'

    # 检查文件是否存在
    if not X_path.exists() or not y_path.exists():
        print('Training files not found. Expected:', X_path, y_path)
        return

    # 检查并安装必要的依赖
    ensure_deps()

    # 导入机器学习库
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    import numpy as np
    from datetime import datetime

    # 读取训练数据
    X_rows = read_jsonl(X_path)
    y_rows = read_jsonl(y_path)
    
    if not X_rows:
        print('No training rows found in', X_path)
        return

    # 将标签从字典格式转换为整数列表
    # y_rows 中的每个元素是 {'label': 0 或 1} 的格式
    y = [int(r.get('label', 0)) if isinstance(r, dict) else int(r) for r in y_rows]

    # ============================================================================
    # 步骤1：数据分割 - 将数据分为训练集、验证集、测试集
    # ============================================================================
    test_size = 0.15  # 测试集占比 15%
    val_size = 0.15   # 验证集占比 15%
    temp_size = test_size + val_size  # 临时集 = 验证集 + 测试集

    # 尝试使用分层抽样保持每个分割中的类别比例
    strat = None
    try:
        # 只有在数据包含多个类别时才进行分层
        if len(set(y)) > 1:
            strat = y
    except Exception:
        strat = None

    # 第一次分割：训练集 vs 临时集(验证+测试)
    X_train_rows, X_temp, y_train, y_temp = train_test_split(
        X_rows, y, test_size=temp_size, random_state=42, stratify=strat
    )
    
    # 第二次分割：从临时集中分离验证集和测试集
    if temp_size > 0 and len(X_temp) >= 2:
        val_frac = val_size / temp_size
        strat_temp = None
        try:
            if len(set(y_temp)) > 1:
                strat_temp = y_temp
        except Exception:
            strat_temp = None
        
        # 如果临时集样本过少（< 2），全部用作验证集
        if len(X_temp) < 2:
            X_val_rows, X_test_rows, y_val, y_test = X_temp, [], y_temp, []
        else:
            X_val_rows, X_test_rows, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - val_frac), random_state=42, stratify=strat_temp
            )
    else:
        # 如果临时集为空，验证集和测试集也为空
        X_val_rows, X_test_rows, y_val, y_test = [], [], [], []

    # ============================================================================
    # 步骤2：将分割后的数据保存到磁盘，便于后续查看和调试
    # ============================================================================
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    def save_jsonl(rows, path):
        """保存记录列表为 JSONL 格式文件"""
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                json.dump(r, f, ensure_ascii=False)
                f.write('\n')

    # 分别保存各个分割的特征和标签
    save_jsonl(X_train_rows, TRAIN_DIR / 'X_train.jsonl')
    save_jsonl(X_val_rows, TRAIN_DIR / 'X_val.jsonl')
    save_jsonl(X_test_rows, TRAIN_DIR / 'X_test.jsonl')
    save_jsonl([{'label': v} for v in y_train], TRAIN_DIR / 'y_train.jsonl')
    save_jsonl([{'label': v} for v in y_val], TRAIN_DIR / 'y_val.jsonl')
    save_jsonl([{'label': v} for v in y_test], TRAIN_DIR / 'y_test.jsonl')

    print('Saved splits: X_train/y_train, X_val/y_val, X_test/y_test')

    # ============================================================================
    # 步骤3：特征向量化 - 将字典特征转换为数值矩阵
    # ============================================================================
    # DictVectorizer 处理分类特征和数值特征，将字典转换为 sklearn 可用的稀疏或稠密矩阵
    vec = DictVectorizer(sparse=False)  # sparse=False 保存稠密矩阵便于后续处理
    X_train = vec.fit_transform(X_train_rows)  # fit_transform 在训练集上学习特征空间
    
    # transform 验证集和测试集，使用训练集学到的特征空间
    X_val = vec.transform(X_val_rows) if len(X_val_rows) else np.zeros((0, X_train.shape[1]))
    X_test = vec.transform(X_test_rows) if len(X_test_rows) else np.zeros((0, X_train.shape[1]))

    # ============================================================================
    # 步骤4：模型训练 - 训练随机森林分类器
    # ============================================================================
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # ============================================================================
    # 步骤5：概率校准 - 提高预测概率的可靠性
    # ============================================================================
    # CalibratedClassifierCV 使用验证集对模型的概率预测进行校准
    # 这确保 predict_proba() 返回的概率更加准确和可靠
    calibrated_clf = None
    try:
        from sklearn.calibration import CalibratedClassifierCV
        if len(X_val) and len(y_val):
            # 'prefit' 模式表示使用已经训练好的分类器
            # 'sigmoid' 方法使用 Platt 标度进行校准
            try:
                calibrated = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
                calibrated.fit(X_val, y_val)
                calibrated_clf = calibrated
            except Exception:
                # 如果校准失败，继续使用未校准的模型
                calibrated_clf = None
    except Exception:
        calibrated_clf = None

    # ============================================================================
    # 步骤6：模型评估 - 在验证集和测试集上评估性能
    # ============================================================================
    results = {}
    
    # 验证集评估
    if len(X_val) and len(y_val):
        y_pred_val = clf.predict(X_val)
        # 计算准确率：正确预测数 / 总预测数
        results['val_acc'] = accuracy_score(y_val, y_pred_val)
        # 生成详细的分类报告（精准率、召回率、F1分数等）
        results['val_report'] = classification_report(y_val, y_pred_val)
    
    # 测试集评估
    if len(X_test) and len(y_test):
        y_pred_test = clf.predict(X_test)
        results['test_acc'] = accuracy_score(y_test, y_pred_test)
        results['test_report'] = classification_report(y_test, y_pred_test)

    # ============================================================================
    # 步骤7：保存模型和向量化器
    # ============================================================================
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'model.joblib'
    vec_path = MODEL_DIR / 'vectorizer.joblib'

    # 选择要保存的模型：如果校准成功则保存校准模型，否则保存原始模型
    model_to_save = calibrated_clf if calibrated_clf is not None else clf

    # 使用 joblib 序列化并保存模型对象到磁盘
    joblib.dump(model_to_save, str(model_path))
    joblib.dump(vec, str(vec_path))

    print('Trained RandomForest model saved to', model_path)
    print('Vectorizer saved to', vec_path)
    if 'val_acc' in results:
        print('Validation accuracy:', results['val_acc'])
        print('Validation report:\n', results['val_report'])
    if 'test_acc' in results:
        print('Test accuracy:', results['test_acc'])
        print('Test report:\n', results['test_report'])

    # ============================================================================
    # 步骤8：保存训练报告（英文版本）
    # ============================================================================
    # 创建详细的训练报告记录此次训练的各种参数和结果
    logs_dir = ROOT / 'logs' / 'training'
    logs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report = {
        'timestamp': now,
        'model_path': str(model_path),
        'vectorizer_path': str(vec_path),
        'train_size': len(X_train_rows),
        'val_size': len(X_val_rows),
        'test_size': len(X_test_rows),
        'results': results,
    }
    
    report_path = logs_dir / f'training_report_{now}.json'
    try:
        with open(report_path, 'w', encoding='utf-8') as rf:
            json.dump(report, rf, ensure_ascii=False, indent=2)
        print('Saved training report to', report_path)
    except Exception as e:
        print('Failed to write training report:', e)

    # ============================================================================
    # 步骤9：生成中文版本训练报告
    # ============================================================================
    # 为中文用户提供本地化的报告
    try:
        from src.i18n.zh_cn import translate_training_report
        zh_report = translate_training_report(report)
        zh_report_path = logs_dir / f'training_report_{now}_zh.json'
        with open(zh_report_path, 'w', encoding='utf-8') as rf:
            json.dump(zh_report, rf, ensure_ascii=False, indent=2)
        print('Saved Chinese training report to', zh_report_path)
    except Exception as e:
        print('Warning: Could not generate Chinese report:', e)

    # ============================================================================
    # 步骤10：更新模型注册表
    # ============================================================================
    # 模型注册表维护一个所有已训练模型的中央记录
    # 便于模型版本管理、追溯和管理
    try:
        registry_path = ROOT / 'data' / 'models' / 'registry.json'
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取现有注册表（如果存在）
        registry = []
        if registry_path.exists():
            try:
                with open(registry_path, 'r', encoding='utf-8') as rf:
                    registry = json.load(rf) or []
            except Exception:
                registry = []
        
        # 添加新的模型条目
        entry = {
            'timestamp': now,
            'model_path': str(model_path),
            'vectorizer_path': str(vec_path),
            'train_size': len(X_train_rows),
            'val_size': len(X_val_rows),
            'test_size': len(X_test_rows),
            'results': results,
        }
        registry.append(entry)
        with open(registry_path, 'w', encoding='utf-8') as rf:
            json.dump(registry, rf, ensure_ascii=False, indent=2)
        print('Updated model registry at', registry_path)
    except Exception as e:
        print('Failed to update registry:', e)


if __name__ == '__main__':
    """
    直接运行此脚本时的入口
    
    使用示例：
        python src/models/trainers/train_baseline.py
    """
    run()

if __name__ == '__main__':
    run()
