#!/usr/bin/env python3
"""
Railway部署专用启动脚本
处理云平台环境变量和配置
"""

import os
import uvicorn
from simple_server import app

if __name__ == "__main__":
    # 获取Railway环境变量
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 启动MoldingFlow后端服务...")
    print(f"📍 监听地址: {host}:{port}")
    print(f"🔗 API文档: http://{host}:{port}/docs")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )