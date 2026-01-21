"""
特征提取模块 - 从解析后的 Nginx 日志中提取机器学习特征

本模块从结构化的日志记录中提取和转换特征，用于训练机器学习模型。
包括以下功能：
1. 请求解析 - 从 HTTP 请求字符串中提取方法、路径、扩展名
2. 时间特征提取 - 从时间戳中提取小时、周几等特征
3. 特征工程 - 计算派生特征，处理缺失值
4. 数据保存 - 支持多种格式 (CSV, JSON, Parquet) 保存特征
5. 地理位置和用户代理信息处理

作者: nginx-log-ai-system
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# 项目根目录路径
ROOT = Path(__file__).resolve().parents[2]
# 解析后日志的存储目录
RAW_PARSED_DIR = ROOT / 'data' / 'processed' / 'parsed_logs'
# 提取特征的输出目录
OUT_DIR = ROOT / 'data' / 'processed' / 'features'


def parse_request(request: str):
    """
    解析 HTTP 请求字符串，提取关键信息
    
    从类似 "GET /path/to/resource?query=value HTTP/1.1" 的请求字符串中
    提取 HTTP 方法、请求路径和文件扩展名等信息。
    
    参数:
        request: HTTP 请求字符串，格式为 "METHOD PATH [HTTP_VERSION]"
    
    返回:
        tuple: (method, path, ext)
            - method: HTTP 请求方法 (GET, POST, PUT, DELETE 等)
            - path: 请求的完整路径（包括查询参数）
            - ext: 文件扩展名（如 'jpg', 'html'），如果无扩展名则为 None
    
    示例:
        >>> parse_request("GET /api/users?id=123 HTTP/1.1")
        ('GET', '/api/users', None)
        >>> parse_request("POST /upload/image.jpg HTTP/1.1")
        ('POST', '/upload/image.jpg', 'jpg')
    """
    if not request:
        return None, None, None
    
    parts = request.split()
    if len(parts) < 2:
        return None, None, None
    
    method = parts[0]  # HTTP 方法
    url = parts[1]      # 请求 URL
    http_ver = parts[2] if len(parts) > 2 else None  # HTTP 版本
    
    # 从 URL 中分离出路径（去掉查询参数）
    path = url.split('?')[0]
    
    # 从路径的文件名中提取扩展名
    ext = None
    if '.' in Path(path).name:
        ext = Path(path).suffix.lower().lstrip('.')
    
    return method, path, ext


def extract_features_from_record(rec: dict):
    """
    从日志记录中提取所有特征
    
    将一条完整的日志记录转换为机器学习所需的特征字典。
    处理数据类型转换、时间特征计算、缺失值处理等。
    
    参数:
        rec: 包含日志信息的字典，包括：
            - remote_addr: 客户端 IP
            - status: HTTP 响应状态码
            - body_bytes_sent: 响应体字节数
            - http_referer: 来源页面
            - http_user_agent: 用户代理字符串
            - timestamp/time_local: 请求时间戳
            - request: 完整的 HTTP 请求行
            - geo_*: 地理位置信息（可选）
            - browser: 浏览器信息（可选）
            - os: 操作系统信息（可选）
    
    返回:
        dict: 提取的特征字典，包含以下字段：
            - remote_addr: 客户端 IP
            - status: HTTP 状态码
            - body_bytes_sent: 响应体大小
            - http_referer: HTTP 来源
            - http_user_agent: 用户代理
            - timestamp: 原始时间戳
            - hour: 请求时间的小时（0-23）
            - day_of_week: 请求时间的周几（0-6，0 为周一）
            - method: HTTP 方法
            - path: 请求路径
            - extension: 文件扩展名
            - request_length: 请求字符串长度
            - geo_country: 地理位置国家代码
            - geo_city: 地理位置城市
            - geo_latitude: 地理位置纬度
            - geo_longitude: 地理位置经度
            - browser: 浏览器类型
            - os: 操作系统类型
    """
    out = {}
    
    # 基本字段直接映射
    out['remote_addr'] = rec.get('remote_addr')
    out['status'] = rec.get('status')
    out['body_bytes_sent'] = rec.get('body_bytes_sent')
    out['http_referer'] = rec.get('http_referer')
    out['http_user_agent'] = rec.get('http_user_agent')

    # 时间特征提取
    ts = rec.get('timestamp') or rec.get('time_local')
    out['timestamp'] = ts
    try:
        if ts:
            # 解析时间戳
            if isinstance(ts, str):
                # 尝试 ISO 格式或 Nginx 标准格式
                dt = datetime.fromisoformat(ts) if 'T' in ts else datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z")
            else:
                dt = None
            
            # 提取时间特征
            if dt:
                out['hour'] = dt.hour  # 请求发生的小时
                out['day_of_week'] = dt.weekday()  # 请求发生的周几
            else:
                out['hour'] = None
                out['day_of_week'] = None
        else:
            out['hour'] = None
            out['day_of_week'] = None
    except Exception:
        # 时间解析失败时设为 None
        out['hour'] = None
        out['day_of_week'] = None

    # 解析 HTTP 请求字符串
    method, path, ext = parse_request(rec.get('request'))
    out['method'] = method
    out['path'] = path
    out['extension'] = ext
    out['request_length'] = len(rec.get('request') or '')

    # 地理位置字段（如果存在）
    out['geo_country'] = rec.get('geo_country')
    out['geo_city'] = rec.get('geo_city')
    out['geo_latitude'] = rec.get('geo_latitude')
    out['geo_longitude'] = rec.get('geo_longitude')
    
    # 用户代理字段（如果存在）
    out['browser'] = rec.get('browser')
    out['os'] = rec.get('os')

    return out


def extract_from_file(path: Path):
    """
    从 JSONL 文件中提取所有特征记录
    
    逐行读取包含日志记录的 JSONL 文件，为每条记录提取特征。
    能够处理格式错误或缺失字段的情况，保证数据管道的健壮性。
    
    参数:
        path: 包含解析后日志的 JSONL 文件路径
    
    返回:
        list: 特征字典列表，每个字典代表一条日志记录的特征
    """
    features = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                rec = json.loads(line)
            except Exception:
                # 跳过格式错误的行
                continue
            
            # 提取该记录的特征
            feat = extract_features_from_record(rec)
            features.append(feat)
    
    return features


def save_features(features, out_path: Path):
    """
    将提取的特征保存到文件
    
    支持多种输出格式：CSV（用于电子表格分析）、JSONL（用于流处理）、
    Parquet（用于大规模数据处理）。根据文件扩展名自动选择格式。
    
    参数:
        features: 特征字典列表
        out_path: 输出文件路径，扩展名决定保存格式
            - .csv: CSV 格式
            - .jsonl: JSON Lines 格式
            - .parquet: Apache Parquet 格式
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 尝试导入 pandas 用于高效的数据操作
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd:
        # 使用 pandas 处理数据
        df = pd.DataFrame(features)
        # try parquet first
        try:
            df.to_parquet(str(out_path.with_suffix('.parquet')))
            return str(out_path.with_suffix('.parquet'))
        except Exception:
            # fallback to csv
            df.to_csv(str(out_path.with_suffix('.csv')), index=False)
            return str(out_path.with_suffix('.csv'))
    else:
        # write jsonl
        outp = out_path.with_suffix('.jsonl')
        with open(outp, 'w', encoding='utf-8') as f:
            for r in features:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        return str(outp)


def run(max_files: int = None):
    files = sorted([p for p in RAW_PARSED_DIR.iterdir() if p.is_file()])
    if max_files:
        files = files[:max_files]
    total = 0
    outputs = []
    for p in files:
        feats = extract_from_file(p)
        out_file = OUT_DIR / f'features_{p.stem}'
        saved = save_features(feats, out_file)
        print(f'Processed {p.name}: {len(feats)} rows -> {saved}')
        total += len(feats)
        outputs.append((p.name, len(feats), saved))
    print('Total features extracted:', total)
    return outputs


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--max-files', type=int, default=None)
    args = ap.parse_args()
    run(max_files=args.max_files)
