"""
FastAPI 应用程序 - Nginx 日志 AI 系统 REST API 服务

本模块实现了一个 RESTful API，用于对 Nginx 日志进行实时预测和异常检测。
主要功能包括：
1. 模型加载和管理 - 从磁盘加载预训练的随机森林模型和向量化器
2. 健康检查 - /health 端点用于服务可用性监控
3. 预测服务 - /predict 端点接收日志特征并返回异常预测
4. API 认证 - 支持 X-Api-Key 头部的可选认证
5. 日志记录 - 将所有预测结果保存到 JSONL 文件供审计和分析

作者: nginx-log-ai-system
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from pathlib import Path
import joblib
import os
import time
import json
from typing import Any, Dict, Optional

# 项目根目录路径（相对于当前文件向上两级）
ROOT = Path(__file__).resolve().parents[2]
# 已训练模型所在目录
MODEL_DIR = ROOT / 'data' / 'models' / 'trained_models' / 'random_forest'


class PredictRequest(BaseModel):
    """
    预测请求数据模型
    
    包含客户端向服务器发送的日志特征信息，用于进行异常检测预测。
    
    属性:
        features: 包含日志特征的字典，例如：
            - remote_addr: 客户端IP地址
            - request_method: HTTP请求方法 (GET, POST等)
            - request_path: 请求路径
            - status: HTTP响应状态码
            - body_bytes_sent: 响应体大小（字节数）
            等更多特征...
    """
    features: Dict[str, Any]

    class Config:
        # 提供 API 文档中的示例数据
        schema_extra = {
            "example": {
                "features": {
                    "remote_addr": "127.0.0.1",
                    "request_method": "GET",
                    "request_path": "/",
                }
            }
        }



# 初始化 FastAPI 应用
app = FastAPI(title='nginx-log-ai-system API')


def check_api_key(x_api_key: Optional[str] = Header(None)):
    """
    API 密钥认证依赖函数
    
    验证客户端提供的 API 密钥是否有效。如果环境变量中未设置 API_KEY，
    则允许匿名访问；否则要求提供有效的密钥。
    
    参数:
        x_api_key: 从 HTTP 请求头 'X-Api-Key' 中获取的密钥值
    
    返回:
        bool: 认证通过返回 True
    
    异常:
        HTTPException: 密钥无效或缺失时抛出 401 错误
    """
    expected = os.environ.get('API_KEY')
    if expected:
        if x_api_key is None or x_api_key != expected:
            raise HTTPException(status_code=401, detail='Invalid or missing API key')
    return True


def load_artifacts():
    """
    加载预训练模型和特征向量化器
    
    从磁盘读取已保存的机器学习模型（随机森林）和特征向量化器，
    同时尝试读取模型注册表中的元数据信息。
    
    返回:
        tuple: (clf, vec, model_info)
            - clf: 训练好的随机森林分类器对象
            - vec: DictVectorizer 特征向量化器
            - model_info: 从注册表读取的模型元数据（如果可用）
    
    异常:
        FileNotFoundError: 当模型文件或向量化器文件不存在时抛出
    """
    # 模型和向量化器的保存路径
    model_path = MODEL_DIR / 'model.joblib'
    vec_path = MODEL_DIR / 'vectorizer.joblib'
    
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError('Model or vectorizer not found')
    
    # 使用 joblib 加载序列化的 Python 对象
    clf = joblib.load(str(model_path))
    vec = joblib.load(str(vec_path))
    
    # 尝试从模型注册表读取模型的元数据信息
    registry_path = ROOT / 'data' / 'models' / 'registry.json'
    model_info = None
    try:
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as rf:
                reg = json.load(rf)
                # 注册表是一个数组，取最后一个条目（最新的模型）
                if reg:
                    model_info = reg[-1]
    except Exception:
        # 如果读取失败，继续运行但不记录元数据
        model_info = None
    
    return clf, vec


@app.on_event('startup')
def startup_event():
    """
    FastAPI 启动事件处理函数
    
    当应用启动时调用此函数，加载预训练模型和特征向量化器。
    如果加载失败，将错误信息保存到应用状态，但应用仍继续运行
    以便提供诊断信息（/health 端点会返回错误详情）。
    """
    try:
        # 加载模型和向量化器
        app.state.clf, app.state.vec = load_artifacts()
    except Exception as e:
        # 在加载失败时设置相关错误状态
        # 应用会继续运行但预测端点会返回 503 服务不可用错误
        app.state.clf, app.state.vec = None, None
        app.state.load_error = str(e)


@app.get('/health')
def health():
    """
    健康检查端点
    
    返回服务的健康状态。用于容器编排系统（如 Kubernetes）或
    负载均衡器检测服务是否正常运行。
    
    返回:
        dict: 包含状态信息
            - status: 'ok' 表示服务正常，'unhealthy' 表示模型加载失败
            - error: (可选) 错误信息描述
    """
    if getattr(app.state, 'clf', None) is None:
        return {'status': 'unhealthy', 'error': getattr(app.state, 'load_error', 'model not loaded')}
    return {'status': 'ok'}


@app.post('/predict')
def predict(req: PredictRequest, authorized: bool = Depends(check_api_key)):
    """
    异常检测预测端点
    
    接收日志特征数据，使用训练好的随机森林模型进行实时预测，
    判断该请求是否为异常行为。
    
    参数:
        req: PredictRequest 对象，包含日志特征
        authorized: API 认证结果（通过 check_api_key 依赖注入）
    
    返回:
        dict: 预测结果
            - prediction: 预测标签（0=正常，1=异常）
            - probability: 预测为异常类的概率值（0-1）
    
    异常:
        HTTPException: 模型未加载时返回 503 服务不可用
    """
    # 检查模型是否已成功加载
    if getattr(app.state, 'clf', None) is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    try:
        # 获取请求特征
        x = req.features
        
        # 使用向量化器将特征字典转换为稀疏矩阵
        X = app.state.vec.transform([x])
        
        # 进行预测
        pred = app.state.clf.predict(X)
        
        # 尝试获取预测的概率（如果模型支持）
        prob = None
        try:
            # 获取预测为正常类的概率，取第二列（异常类）
            prob = float(app.state.clf.predict_proba(X)[:, 1][0])
        except Exception:
            # 某些模型可能不支持概率预测
            prob = None

        # 构建响应对象
        out = {'prediction': int(pred[0]), 'probability': prob}

        # 将预测记录到日志文件用于审计、监控和后续分析
        try:
            log_dir = ROOT / 'logs' / 'predictions'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / 'predictions.jsonl'
            
            # 构建日志条目，包含时间戳、特征和预测结果
            entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'features': x,
                'prediction': int(pred[0]),
                'probability': prob,
            }
            with open(log_path, 'a', encoding='utf-8') as lf:
                lf.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception:
            pass

        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
