"""
训练数据准备模块 - 为机器学习模型构建训练集

本模块从处理后的特征中：
1. 加载特征配置 - 确定哪些特征参与模型训练
2. 特征选择 - 根据配置选择数值和分类特征
3. 数据标签化 - 为特征添加监督学习标签
4. 数据保存 - 将特征和标签保存为 JSONL 格式供模型训练

输出数据格式：
- X_train.jsonl: 特征记录（字典，每行一个）
- y_train.jsonl: 标签记录（{label: 0/1}，每行一个）

标签规则：HTTP 状态码 >= 400 -> 异常（1），否则 -> 正常（0）

作者: nginx-log-ai-system
"""

import sys
from pathlib import Path
import json

# 项目根目录路径
ROOT = Path(__file__).resolve().parents[2]
# 处理后特征存储目录
FEATURE_DIR = ROOT / 'data' / 'processed' / 'features'
# 训练数据输出目录
TRAIN_DIR = ROOT / 'data' / 'processed' / 'training' / 'train'
# 特征配置文件路径
CFG_PATH = ROOT / 'configs' / 'feature_config.yaml'


def load_feature_config(cfg_path: Path):
    """
    从配置文件加载特征定义
    
    读取 YAML 格式的特征配置文件，定义了每个特征的元数据，
    包括特征名称、数据类型、在模型中的角色等。
    
    参数:
        cfg_path: 配置文件路径
    
    返回:
        dict: 包含特征定义的配置字典，格式如下：
            {
                'features': [
                    {'name': 'status', 'role': 'numeric', ...},
                    {'name': 'method', 'role': 'categorical', ...},
                    ...
                ]
            }
        如果文件不存在或无法解析，返回 None
    
    注:
        支持两种解析方式：
        1. 使用 PyYAML 库（如果已安装）
        2. 简单的文本解析（作为回退方案）
    """
    if not cfg_path.exists():
        return None
    
    try:
        import yaml
    except Exception:
        yaml = None
    
    if yaml:
        # 使用 PyYAML 解析 YAML 文件
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    # 回退：简单的文本解析方式
    # 预期格式：
    # - name: feature_name
    #   role: numeric/categorical
    features = []
    cur = None
    with open(cfg_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.strip().startswith('- name:'):
                # 新特征条目
                name = line.split(':', 1)[1].strip()
                cur = {'name': name}
                features.append(cur)
            elif ':' in line and cur is not None:
                # 特征属性
                k, v = line.split(':', 1)
                k = k.strip().lstrip('- ')
                v = v.strip()
                cur[k] = v or None
    
    return {'features': features}


def find_processed_files(feature_dir: Path):
    """
    查找已处理的特征文件
    
    扫描特征目录，找到所有以 'processed_' 开头的特征文件。
    支持多种格式（Parquet、CSV、JSONL）。
    
    参数:
        feature_dir: 特征目录路径
    
    返回:
        list: 已排序的特征文件路径列表
    """
    files = []
    for p in feature_dir.iterdir():
        if p.is_file() and p.name.startswith('processed_'):
            files.append(p)
    
    # 如果未找到，尝试寻找包含 'processed' 的文件
    if not files:
        for p in feature_dir.iterdir():
            if p.is_file() and 'processed' in p.name:
                files.append(p)
    
    return sorted(files)


def read_records(path: Path):
    """
    从特征文件读取记录
    
    支持多种文件格式，自动根据扩展名选择合适的读取方式。
    优先使用 pandas 库进行高效读取。
    
    参数:
        path: 特征文件路径
    
    返回:
        list: 记录列表，每个记录是一个字典
    """
    try:
        import pandas as pd
    except Exception:
        pd = None
    
    # 尝试使用 pandas 读取 Parquet 格式
    if pd and path.suffix == '.parquet':
        df = pd.read_parquet(str(path))
        return df.to_dict(orient='records')
    
    # 尝试使用 pandas 读取 CSV 格式
    if pd and path.suffix == '.csv':
        df = pd.read_csv(str(path))
        return df.to_dict(orient='records')
    
    # 回退：读取 JSONL 格式
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    
    return rows


def label_record(rec: dict) -> int:
    """
    为记录分配标签（监督学习）
    
    基于 HTTP 响应状态码对日志记录进行分类：
    - 状态码 >= 400: 异常（1）
    - 状态码 < 400: 正常（0）
    
    参数:
        rec: 特征记录字典
    
    返回:
        int: 标签值（0 表示正常，1 表示异常）
    """
    s = rec.get('status')
    try:
        s = int(s)
    except Exception:
        return 0
    
    # HTTP 4xx 和 5xx 被视为异常
    return 1 if s >= 400 else 0


def select_feature_columns(rec: dict, cfg: dict):
    """
    根据配置从记录中选择特征列
    
    从完整的特征记录中提取用于模型训练的特征。
    只选择在配置文件中标记为 'numeric' 或 'categorical' 角色的特征。
    
    参数:
        rec: 特征记录字典
        cfg: 特征配置字典
    
    返回:
        list: 选中的特征列名列表
    """
    if not cfg:
        # 默认特征选择（如果无配置文件）
        keys = [
            'status',           # HTTP 状态码
            'body_bytes_sent',  # 响应体大小
            'hour',             # 请求时间小时
            'request_length',   # 请求长度
            'geo_latitude',     # 地理位置纬度
            'geo_longitude',    # 地理位置经度
            'method',           # HTTP 方法
            'extension',        # 文件扩展名
            'geo_country',      # 地理位置国家
            'browser',          # 浏览器类型
            'os'                # 操作系统类型
        ]
        return [k for k in keys if k in rec]
    
    # 根据配置文件选择特征
    feat_defs = cfg.get('features', [])
    cols = []
    for f in feat_defs:
        name = f.get('name')
        # 只选择标记为 numeric 或 categorical 的特征
        if name in rec and f.get('role') in ('numeric', 'categorical'):
            cols.append(name)
    
    return cols


def run():
    """
    主函数 - 执行完整的训练数据准备流程
    
    流程：
    1. 查找已处理的特征文件
    2. 加载特征配置
    3. 遍历所有特征文件
    4. 为每条记录选择特征和标签
    5. 保存特征和标签到 JSONL 文件
    
    输出：
    - X_train.jsonl: 特征数据
    - y_train.jsonl: 标签数据
    """
    # 查找已处理的特征文件
    files = find_processed_files(FEATURE_DIR)
    if not files:
        print('No processed feature files found in', FEATURE_DIR)
        return

    # 加载特征配置
    cfg = load_feature_config(CFG_PATH)
    
    all_X = []
    all_y = []
    total = 0
    
    # 处理每个特征文件
    for p in files:
        recs = read_records(p)
        for r in recs:
            # 选择该记录的特征
            cols = select_feature_columns(r, cfg)
            x = {c: r.get(c) for c in cols}
            
            # 为该记录分配标签
            y = label_record(r)
            
            all_X.append(x)
            all_y.append({'label': y})
        
        print(f'Loaded {len(recs)} records from {p.name}')
        total += len(recs)

    # 创建输出目录
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存特征和标签到 JSONL 格式
    x_path = TRAIN_DIR / 'X_train.jsonl'
    y_path = TRAIN_DIR / 'y_train.jsonl'
    with open(x_path, 'w', encoding='utf-8') as fx, open(y_path, 'w', encoding='utf-8') as fy:
        for xi, yi in zip(all_X, all_y):
            # 每行一个 JSON 对象
            fx.write(json.dumps(xi, ensure_ascii=False) + '\n')
            fy.write(json.dumps(yi, ensure_ascii=False) + '\n')

    print('Wrote X_train ->', x_path)
    print('Wrote y_train ->', y_path)
    print('Total rows:', total)


if __name__ == '__main__':
    run()
