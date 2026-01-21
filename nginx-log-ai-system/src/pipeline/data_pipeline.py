"""
数据处理管道 - 完整的 Nginx 日志处理流程

本模块实现了从原始 Nginx 日志到结构化数据的完整处理流程：
1. 数据读取 - 从 Nginx 日志文件读取行
2. 数据清理 - 解析日志格式，处理错误数据
3. 数据标准化 - 统一时间戳格式
4. 数据富化 - 添加地理位置、用户代理等信息
5. 数据存储 - 保存为 JSON Lines 格式或数据库

支持的功能：
- 地理位置查询（GeoIP）
- 用户代理解析（浏览器、操作系统识别）
- 可选的数据库持久化
- 可配置的处理范围（最大行数）

作者: nginx-log-ai-system
"""

import sys
from pathlib import Path
import argparse

# 项目根目录路径
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入数据处理模块
from src.data.collector import read_lines
from src.data.cleaner.data_cleaner import clean_line
from src.data.cleaner.normalizer import normalize_timestamp
from src.data.cleaner.enricher import GeoIPEnricher
from src.features.utils.user_agent import parse_user_agent
from src.data.storage.file_storage import save_json_lines
from src.data.storage import database

# 数据目录配置
RAW_DIR = ROOT / 'data' / 'raw' / 'nginx_logs'  # 原始日志输入目录
OUT_DIR = ROOT / 'data' / 'processed' / 'parsed_logs'  # 处理后日志输出目录
DB_PATH = ROOT / 'data' / 'models' / 'logs.db'  # SQLite 数据库路径


def process_file(path: Path, max_lines: int = None, to_db: bool = False):
    """
    处理单个日志文件
    
    对单个 Nginx 日志文件执行完整的处理流程：
    1. 逐行读取日志
    2. 解析日志格式
    3. 清理和标准化数据
    4. 添加地理位置信息
    5. 解析用户代理
    6. 保存处理结果
    
    参数:
        path: 日志文件路径
        max_lines: 最多处理的行数（None 表示处理全部）
        to_db: 是否将结果保存到数据库
    
    返回:
        tuple: (记录数, 输出文件路径)
    
    示例:
        >>> count, out_path = process_file(Path('access.log'), max_lines=10000)
        >>> print(f'Processed {count} records')
    """
    records = []
    
    # 初始化地理位置富化器
    enricher = GeoIPEnricher(ROOT)
    
    # 逐行读取和处理日志
    for line in read_lines(str(path), max_lines=max_lines):
        # 第一步：解析日志格式
        rec = clean_line(line)
        if not rec:
            continue  # 跳过无法解析的行
        
        # 第二步：标准化时间戳
        rec = normalize_timestamp(rec)
        
        # 第三步：添加地理位置信息
        # 根据客户端 IP 查询 GeoIP 数据库，添加国家、城市、坐标等信息
        rec = enricher.enrich(rec)
        
        # 第四步：解析用户代理字符串
        # 识别客户端浏览器、操作系统等信息
        ua = rec.get('http_user_agent')
        ua_info = parse_user_agent(ua)
        rec.update(ua_info)
        
        records.append(rec)

    # 创建输出目录
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件路径
    out_path = OUT_DIR / f'parsed_{path.name}.jsonl'
    
    # 将处理后的记录保存为 JSON Lines 格式
    save_json_lines(records, str(out_path))

    # 可选：将记录持久化到 SQLite 数据库
    if to_db:
        conn = database.init_db(str(DB_PATH))
        for r in records:
            database.insert_log(conn, r)
        conn.close()

    return len(records), out_path


def process_all(max_lines: int = None, to_db: bool = False):
    """
    处理日志目录中的所有文件
    
    查找目录中的所有日志文件并依次处理。生成处理摘要和统计信息。
    
    参数:
        max_lines: 每个文件最多处理的行数
        to_db: 是否将结果保存到数据库
    
    返回:
        tuple: (总记录数, 处理结果列表)
            处理结果列表中每个元素为 (文件名, 记录数, 输出路径)
    """
    # 查找所有日志文件
    files = [p for p in RAW_DIR.iterdir() if p.is_file()]
    
    total = 0
    outputs = []
    
    # 处理每个日志文件
    for f in files:
        n, out_path = process_file(f, max_lines=max_lines, to_db=to_db)
        print(f'Processed {f.name}: {n} records -> {out_path}')
        total += n
        outputs.append((f.name, n, out_path))
    
    print('Total records processed:', total)
    return total, outputs


def main():
    """
    命令行入口函数
    
    支持的命令行参数：
    - --max-lines: 每个文件最多处理的行数（默认：无限制）
    - --to-db: 是否将结果保存到 SQLite 数据库（布尔标志）
    
    示例：
        python data_pipeline.py --max-lines 10000 --to-db
    """
    parser = argparse.ArgumentParser(description='Data pipeline for nginx logs')
    parser.add_argument('--max-lines', type=int, default=None, 
                        help='Max lines to process per file')
    parser.add_argument('--to-db', action='store_true', dest='to_db', 
                        help='Persist parsed records into sqlite DB')
    args = parser.parse_args()

    # 执行数据处理
    process_all(max_lines=args.max_lines, to_db=args.to_db)


if __name__ == '__main__':
    main()
