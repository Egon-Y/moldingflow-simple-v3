#!/usr/bin/env python3
"""
Railway部署专用启动脚本
处理云平台环境变量和端口配置
"""

import os
import uvicorn
from simple_server import app

def main():
    # 获取Railway环境变量
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 启动MoldingFlow AI API服务器...")
    print(f"📡 监听地址: {host}:{port}")
    print(f"🔗 API文档: http://{host}:{port}/docs")
    print(f"🏥 健康检查: http://{host}:{port}/health")
    
    # 启动服务器
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()