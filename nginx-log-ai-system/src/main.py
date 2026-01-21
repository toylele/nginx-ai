"""
主程序入口 - nginx-log-ai-system

项目说明：
---------
Nginx Log AI System 是一个端到端的机器学习系统，用于分析和预测 Nginx web 服务器的日志。
该系统能够识别异常访问模式和安全威胁。

核心组件：
1. 数据处理管道 (data_pipeline) - 解析和清理原始日志
2. 特征提取 (feature_extractor) - 从日志中提取ML特征
3. 模型训练 (train) - 使用随机森林或其他算法训练模型
4. 模型评估 (evaluate) - 验证模型性能
5. REST API (api) - 提供实时预测服务
6. 报表生成 (report) - 生成分析报告

工作流：
原始日志 -> 数据清理 -> 特征提取 -> 特征标准化 -> 模型训练 -> 模型评估
    |                                                           |
    +--- 预测 API <- 模型部署 <- 性能验证 <- 超参数调优 <--------+

使用方式：
1. 准备原始 Nginx 日志文件到 data/raw/nginx_logs/
2. 运行数据处理管道进行数据清理和富化
3. 提取特征并准备训练数据
4. 训练和评估模型
5. 使用 API 进行在线预测或生成分析报告

作者: nginx-log-ai-system Team
版本: 1.0
"""


def main():
    """
    主函数 - 打印项目信息
    
    当直接运行此模块时，输出项目名称。
    实际的处理流程应通过调用特定的子模块完成。
    """
    print('nginx-log-ai-system')
    print('Nginx 日志异常检测 AI 系统')


if __name__ == '__main__':
    main()
