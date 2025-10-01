# MoldingFlow AI 系统云平台部署指南

## 部署架构
- **前端**: Vercel（静态文件托管）
- **后端**: Railway（Python API服务）
- **优势**: 免费额度、自动HTTPS、全球CDN、自动部署

## 部署步骤

### 1. 准备GitHub仓库
```bash
# 初始化Git仓库
git init
git add .
git commit -m "初始提交: MoldingFlow AI系统"

# 推送到GitHub
git remote add origin https://github.com/你的用户名/moldingflow.git
git push -u origin main
```

### 2. Vercel部署前端

1. 访问 [vercel.com](https://vercel.com)
2. 使用GitHub账号登录
3. 点击"New Project"
4. 导入你的GitHub仓库
5. 配置设置：
   - Framework Preset: Other
   - Root Directory: frontend
6. 点击"Deploy"

**部署完成后获得**：`https://moldingflow.vercel.app`

### 3. Railway部署后端

1. 访问 [railway.app](https://railway.app)
2. 使用GitHub账号登录
3. 点击"New Project"
4. 选择"Deploy from GitHub repo"
5. 选择你的仓库
6. Railway会自动检测Python项目并部署

**部署完成后获得**：`https://moldingflow-backend.railway.app`

### 4. 配置环境变量

在Railway项目设置中添加环境变量：
```
VITE_API_URL=https://moldingflow-backend.railway.app
```

### 5. 测试部署

前端访问：`https://moldingflow.vercel.app`
后端API：`https://moldingflow-backend.railway.app`

## 自定义域名（可选）

### Vercel自定义域名
1. 在Vercel项目设置中选择"Domains"
2. 添加你的域名（如：moldingflow.ai）
3. 按照指引配置DNS

### Railway自定义域名
1. 在Railway项目设置中选择"Settings"
2. 添加自定义域名
3. 配置CNAME记录

## 监控和维护

### Vercel监控
- 自动性能监控
- 访问统计
- 错误日志

### Railway监控
- 资源使用情况
- 自动扩缩容
- 健康检查

## 费用说明

### 免费额度
- **Vercel**: 100GB带宽/月，无限网站
- **Railway**: 5美元/月免费额度（足够本项目使用）

### 升级选项
- 团队协作功能
- 更高流量限制
- 优先支持

## 故障排除

### 常见问题
1. **部署失败**: 检查requirements.txt依赖
2. **API连接失败**: 验证环境变量配置
3. **静态资源404**: 检查Vercel构建配置

### 支持资源
- [Vercel文档](https://vercel.com/docs)
- [Railway文档](https://docs.railway.app)
- 项目GitHub Issues

## 下一步
1. 按照上述步骤完成部署
2. 测试所有功能是否正常
3. 分享链接给其他人使用
4. 考虑添加用户认证功能