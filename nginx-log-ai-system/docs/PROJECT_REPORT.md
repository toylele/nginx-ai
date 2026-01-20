# Nginx Log AI System — 使用说明与项目报告

版本：示例实现（2026-01-20）

## 一、项目概述
- 本项目实现了一个基于 Nginx 访问日志的端到端示例流水线：解析 → 清洗/增强 → 特征抽取 → 特征处理（编码/归一化）→ 训练基线模型 → 评估并生成报告（含可视化）。
- 设计目标：可复现、模块化、便于扩展（新增特征、替换模型、上线服务）。

## 二、主要目录与关键产物
- 原始日志：data/raw/nginx_logs/access.log
- 解析结果（JSONL）：[data/processed/parsed_logs](data/processed/parsed_logs)
- 抽取的特征：[data/processed/features](data/processed/features)
- 训练集划分：[data/processed/training](data/processed/training)
- 模型与向量器：[data/models/trained_models/random_forest/model.joblib](data/models/trained_models/random_forest/model.joblib) 和 [vectorizer.joblib](data/models/trained_models/random_forest/vectorizer.joblib)
- 训练日志/报告：[logs/training](logs/training)（包含 `*_zh.json` 中文报告）
- 评估指标与图像：[logs/evaluation](logs/evaluation)（包含 `metrics_zh.json` 和 `confusion_matrix.png`）
- 汇总报告（HTML/PDF）：[logs/report/summary_report.html](logs/report/summary_report.html) / [logs/report/summary_report.pdf](logs/report/summary_report.pdf)

（上面为相对路径，工作目录为项目根 `nginx-log-ai-system`）

## 三、先决条件
- Python 3.10+
- 推荐安装依赖（部分脚本会在运行时尝试自动安装）：

```bash
python -m pip install scikit-learn pandas joblib matplotlib pyppeteer
```

## 四、快速使用指南（一键执行流水线）
1. 解析日志（将 `data/raw/nginx_logs/access.log` 解析为 JSONL）：

```bash
python scripts/data/process_nginx_logs.py
```

2. 抽取特征并生成特征文件：

```bash
python src/features/extractor.py
```

3. 应用特征处理（编码/归一化），並寫出處理后的特征與元数据：

```bash
python src/features/processors/processor_runner.py
```

4. 準備訓練數據（生成 `data/processed/training` 下的 X/y 分割）：

```bash
python src/features/prepare_training.py
```

5. 訓練基線模型（隨機森林，自動保存模型與向量器，並寫訓練報告）：

```bash
python src/models/trainers/train_baseline.py
```

6. 評估並生成可視化結果（混淆矩陣、PR 曲線（若模型支持概率））：

```bash
python src/models/evaluators/evaluator.py
```

7. 渲染中文摘要並生成 HTML 報告：

```bash
python scripts/report/render_reports.py
```

8. 將 HTML 轉為 PDF（若本機有 Chrome/Chromium，會復用；否則 pyppeteer 可能會下載 Chromium）：

```bash
python scripts/report/html_to_pdf.py
```

## 五、中文化與報告
- 本項目已添加簡單 i18n 支持（`src/i18n/zh_cn.py`），訓練報告和評估指標會同時輸出中文文件（`*_zh.json`）。
- 匯總報告位於 [logs/report/summary_report.html](logs/report/summary_report.html)（可在瀏覽器打開），或 PDF 版本 [logs/report/summary_report.pdf](logs/report/summary_report.pdf)。

## 六、已知事項與建議
- 小數據集下的評估可能會出現單類樣本警告（混淆矩陣、分類報告），建議使用足夠的歷史日誌以獲得更可靠的評估結果。
- Matplotlib 在默認環境可能不包含中文字體，若圖像注釋中文顯示異常，請在 `src/models/evaluators/evaluator.py` 頂部設置：

```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或系統可用中文字體
matplotlib.rcParams['axes.unicode_minus'] = False
```

- 如果需要便攜的 HTML（包含圖片），可將圖片嵌入為 base64（我可以幫助實現）。

## 七、擴展與部署建議
- 將模型封裝為 REST API（示例位置：`src/api/app.py` — 可按需實現），並使用 Docker 容器化以便生產部署。
- 為生產環境增加：輸入驗證、速率限制、模型版本管理、在線監控（延遲/精度）。
- 在訓練中加入交叉驗證與超參數搜索（`GridSearchCV` / `RandomizedSearchCV`），並記錄最佳模型信息至模型註冊表。

## 八、快速故障排查
- 無法生成 PDF：檢查是否安裝 `wkhtmltopdf` 或確保系統有 Chrome；腳本 `scripts/report/html_to_pdf.py` 會嘗試使用本地 Chrome 或安裝 pyppeteer。
- 找不到解析結果：確認 `data/raw/nginx_logs/access.log` 存在並且腳本具有讀取權限。

## 九、下一步（我可以幫你做）
- 實現 `src/api/app.py` 並生成 Dockerfile 與 `docker-compose.yml`。
- 增強報告樣式並將圖片嵌入為 base64，生成功能更完整的單文件報告。
- 添加 CI 任務自動運行測試並生成報告。

---
如果你希望我把該 MD 轉為漂亮的 PDF（使用自定義 CSS），或把報告合併到 README.md，告訴我偏好樣式或需要補充的信息。 
