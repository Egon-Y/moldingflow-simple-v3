# MoldingFlow AI 系统

基于AI的注塑成型工艺参数优化系统，使用深度学习预测翘曲变形，并通过强化学习优化工艺参数。

## 项目特性

- 🧠 **AI预测**: 使用DNN和GNN模型预测注塑产品翘曲变形
- 🔧 **参数优化**: 基于强化学习的工艺参数自动优化
- 🌐 **Web界面**: 现代化的Web前端界面
- 🚀 **云部署**: 支持Vercel+Railway一键部署
- 📊 **实时分析**: 工艺参数敏感性分析和优化建议

## 技术栈

- **前端**: HTML + JavaScript (Vite)
- **后端**: Python FastAPI
- **AI框架**: PyTorch
- **部署**: Vercel + Railway + Docker

## 快速开始

### 本地开发

1. 安装依赖:
```bash
cd scripts
python install_dependencies.py
```

2. 启动服务:
```bash
cd scripts
python start_server.py
```

3. 访问应用:
- 前端: http://localhost:8080
- 后端API: http://localhost:8000/docs

### 云平台部署

详细部署指南请参考 [DEPLOY_GUIDE.md](./DEPLOY_GUIDE.md)

## 项目结构

```
moldingflow-github-repo/
├── frontend/           # 前端文件
│   ├── index.html     # 主页面
│   ├── vite.config.js # Vite配置
│   └── .env.production # 生产环境配置
├── backend/           # 后端API
│   ├── simple_server.py # FastAPI服务器
│   ├── railway_start.py # Railway部署脚本
│   └── requirements.txt # Python依赖
├── ai_models/        # AI模型
│   ├── predictor.py      # 翘曲预测器
│   ├── rl_optimizer.py   # 强化学习优化器
│   ├── dnn_model.py      # 深度神经网络
│   ├── gnn_model.py      # 图神经网络
│   └── training_pipeline.py # 训练流水线
├── config/           # 配置
│   ├── config.py         # 系统配置
│   ├── docker-compose.yml # Docker编排
│   ├── Dockerfile        # Docker镜像
│   └── .gitignore        # Git忽略规则
├── docs/            # 文档
│   ├── README.md        # 项目说明
│   └── CONTRIBUTING.md  # 贡献指南
├── scripts/         # 脚本
│   ├── start_server.py      # 启动脚本
│   └── install_dependencies.py # 依赖安装
├── vercel.json      # Vercel部署配置
├── railway.json     # Railway部署配置
└── DEPLOY_GUIDE.md  # 部署指南
```

## API接口

- `GET /health` - 健康检查
- `POST /predict` - 翘曲预测
- `POST /optimize` - 参数优化
- `POST /analyze` - 敏感性分析

## 贡献指南

请参考 [CONTRIBUTING.md](./docs/CONTRIBUTING.md)

## 许可证

MIT License
