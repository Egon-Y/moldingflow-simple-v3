from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
import yaml
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MoldingFlow AI API - 简化版")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictPayload(BaseModel):
    temperature: float = 150.0
    pressure: float = 5.0
    time: float = 60.0
    viscosity: float = 1000.0
    cte: float = 15.0

class RecommendPayload(BaseModel):
    search_space: Dict[str, Tuple[float, float]] = {
        "temperature": (80.0, 180.0),
        "pressure": (1.0, 10.0),
        "time": (10.0, 120.0),
    }
    steps: int = 50

class SimpleSurrogateModel:
    """简化的代理模型，使用随机森林"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_training_data(self):
        """生成模拟训练数据"""
        np.random.seed(42)
        n_samples = 1000
        X = np.random.rand(n_samples, 5)  # 5个特征
        X[:, 0] = X[:, 0] * 100 + 80      # 温度 80-180
        X[:, 1] = X[:, 1] * 9 + 1         # 压力 1-10
        X[:, 2] = X[:, 2] * 110 + 10      # 时间 10-120
        X[:, 3] = X[:, 3] * 900 + 100     # 粘度 100-1000
        X[:, 4] = X[:, 4] * 10 + 10       # CTE 10-20
        
        # 模拟翘曲和空洞风险
        y_warpage = (
            0.5 * (X[:, 0] - 130)**2 / 1000 +  # 温度偏离最优值
            0.3 * (X[:, 1] - 6)**2 / 10 +       # 压力偏离最优值
            0.2 * (X[:, 2] - 65)**2 / 1000 +    # 时间偏离最优值
            np.random.normal(0, 0.1, n_samples) # 噪声
        )
        
        y_void = (
            0.4 * (X[:, 1] - 8)**2 / 10 +       # 压力影响空洞
            0.3 * (X[:, 3] - 800)**2 / 10000 + # 粘度影响
            0.3 * (X[:, 0] - 140)**2 / 1000 +   # 温度影响
            np.random.normal(0, 0.05, n_samples) # 噪声
        )
        
        return X, y_warpage, y_void
    
    def train(self):
        """训练模型"""
        X, y_warpage, y_void = self.generate_training_data()
        self.X_scaled = self.scaler.fit_transform(X)
        self.model.fit(self.X_scaled, y_warpage)  # 主要预测翘曲
        self.is_trained = True
        
    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """预测翘曲和空洞风险"""
        if not self.is_trained:
            self.train()
            
        # 提取特征
        features = np.array([[
            payload.get('temperature', 150),
            payload.get('pressure', 5),
            payload.get('time', 60),
            payload.get('viscosity', 1000),
            payload.get('cte', 15)
        ]])
        
        features_scaled = self.scaler.transform(features)
        warpage = float(self.model.predict(features_scaled)[0])
        
        # 简化的空洞风险评估
        void_risk = max(0, min(1, 
            (payload.get('pressure', 5) - 1) / 9 * 0.3 +
            (payload.get('temperature', 150) - 80) / 100 * 0.2 +
            (payload.get('viscosity', 1000) - 100) / 900 * 0.5
        ))
        
        # 生成32x32的伪场数据
        field_data = self.generate_field_data(warpage, void_risk)
        
        return {
            "warpage": round(warpage, 4),
            "void_risk": round(void_risk, 4),
            "warpage_field": field_data["warpage"],
            "void_risk_map": field_data["void"],
            "status": "success"
        }
    
    def generate_field_data(self, warpage: float, void_risk: float) -> Dict[str, List[List[float]]]:
        """生成32x32的伪场数据"""
        size = 32
        warpage_field = []
        void_map = []
        
        for i in range(size):
            warpage_row = []
            void_row = []
            for j in range(size):
                # 基于位置生成变化
                center_dist = ((i - size/2)**2 + (j - size/2)**2) ** 0.5
                warpage_val = warpage * (1 - center_dist/(size*0.7)) + np.random.normal(0, 0.1)
                void_val = void_risk * (1 - center_dist/(size*0.5)) + np.random.normal(0, 0.05)
                
                warpage_row.append(max(0, round(warpage_val, 3)))
                void_row.append(max(0, min(1, round(void_val, 3))))
            
            warpage_field.append(warpage_row)
            void_map.append(void_row)
            
        return {"warpage": warpage_field, "void": void_map}

class SimpleOptimizer:
    """简化的参数优化器"""
    
    def __init__(self):
        self.surrogate = SimpleSurrogateModel()
        
    def optimize(self, search_space: Dict[str, Tuple[float, float]], steps: int = 50) -> Dict[str, Any]:
        """简单网格搜索优化"""
        best_score = float('inf')
        best_params = {}
        
        # 在搜索空间内采样
        for step in range(steps):
            params = {}
            for param, (low, high) in search_space.items():
                params[param] = np.random.uniform(low, high)
                
            # 添加固定参数
            params.update({
                'viscosity': 800,
                'cte': 15
            })
            
            result = self.surrogate.predict(params)
            score = result['warpage'] * 0.6 + result['void_risk'] * 0.4
            
            if score < best_score:
                best_score = score
                best_params = params
                best_result = result
        
        return {
            "params": best_params,
            "score": round(best_score, 4),
            "warpage": best_result['warpage'],
            "void_risk": best_result['void_risk']
        }

# 初始化模型和优化器
surrogate_model = SimpleSurrogateModel()
optimizer = SimpleOptimizer()

@app.get("/")
async def root():
    return {"message": "MoldingFlow AI API - 模压翘曲预测与智能推荐系统"}

@app.post("/predict")
async def predict(payload: PredictPayload):
    """预测翘曲和空洞风险"""
    result = surrogate_model.predict(payload.dict())
    return {"ok": True, "result": result}

@app.post("/recommend")
async def recommend(payload: RecommendPayload):
    """推荐最优工艺参数"""
    best = optimizer.optimize(payload.search_space, payload.steps)
    
    # 生成e-Recipe
    recipe = {
        "temperature_C": round(best["params"]["temperature"], 1),
        "pressure_MPa": round(best["params"]["pressure"], 1),
        "time_s": round(best["params"]["time"], 1),
        "predicted_warpage": best["warpage"],
        "predicted_void_risk": best["void_risk"]
    }
    
    return {
        "ok": True, 
        "best_parameters": best,
        "e_recipe": recipe,
        "optimization_steps": payload.steps
    }

@app.post("/upload-design")
async def upload_design(file: UploadFile = File(...)):
    """上传设计文件（占位功能）"""
    return {
        "ok": True, 
        "filename": file.filename,
        "message": "设计文件上传成功，特征提取完成",
        "design_features": {
            "package_size": "10x10mm",
            "die_count": 1,
            "wire_density": "medium"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_ready": surrogate_model.is_trained}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)