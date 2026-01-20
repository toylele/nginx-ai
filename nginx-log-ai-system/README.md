# nginx-log-ai-system

项目骨架：用于 Nginx 日志的异常检测与分析平台。

## 使用示例：将解析结果持久化到 SQLite

流水线支持将解析后的记录写入 SQLite 数据库，运行示例：

```bash
python -m src.pipeline.data_pipeline --to-db
```

说明：
- 解析输出 JSONL 文件位于 `data/processed/parsed_logs/`。
- 持久化的 SQLite 数据库默认路径：`data/models/logs.db`。
- 测试过程中产生的临时数据库 `data/models/test_logs.db` 已归档至 `data/models/archive/`（按时间戳命名）。

如需限制每个文件处理的行数以便快速测试，可使用 `--max-lines`：

```bash
python -m src.pipeline.data_pipeline --max-lines 1000 --to-db
```

更多使用说明请参阅 docs/ 目录中的文档。
