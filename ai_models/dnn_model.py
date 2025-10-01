"""
深度神经网络模型 - 快速翘曲和空洞预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    """残差块 - 提升深度网络训练效果"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 残差连接
        out = self.activation(out)
        return out

class AttentionLayer(nn.Module):
    """注意力层 - 自动学习特征重要性"""
    
    def __init__(self, input_dim: int):
        super(AttentionLayer, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class MoldingFlowDNN(nn.Module):
    """模压流动深度神经网络"""
    
    def __init__(self,
                 input_dim: int = 10,  # 制程参数 + 几何特征
                 hidden_layers: List[int] = [256, 128, 64, 32],
                 output_dim: int = 2,  # warpage + void_ratio
                 dropout: float = 0.2,
                 use_residual: bool = True,
                 use_attention: bool = True):
        super(MoldingFlowDNN, self).__init__()
        
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 注意力层
        if use_attention:
            self.attention = AttentionLayer(hidden_layers[0])
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        
        for i in range(len(hidden_layers) - 1):
            if use_residual and hidden_layers[i] == hidden_layers[i+1]:
                # 使用残差块（当维度相同时）
                self.hidden_layers.append(ResidualBlock(hidden_layers[i], dropout))
            else:
                # 普通线性层
                self.hidden_layers.append(nn.Sequential(
                    nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                    nn.BatchNorm1d(hidden_layers[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_layers[-1], output_dim),
            nn.ReLU()  # 确保输出为正值
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 输入层
        x = self.input_layer(x)
        
        # 注意力机制
        if self.use_attention:
            x = self.attention(x)
        
        # 隐藏层
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 输出层
        output = self.output_layer(x)
        
        return output

class MultiTaskDNN(nn.Module):
    """多任务深度神经网络 - 同时预测翘曲和空洞"""
    
    def __init__(self,
                 input_dim: int = 10,
                 shared_layers: List[int] = [256, 128],
                 task_layers: List[int] = [64, 32],
                 dropout: float = 0.2):
        super(MultiTaskDNN, self).__init__()
        
        # 共享特征提取器
        shared_modules = []
        prev_dim = input_dim
        
        for hidden_dim in shared_layers:
            shared_modules.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*shared_modules)
        
        # 翘曲预测分支
        warpage_modules = []
        prev_dim = shared_layers[-1]
        
        for hidden_dim in task_layers:
            warpage_modules.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        warpage_modules.append(nn.Linear(prev_dim, 1))
        warpage_modules.append(nn.ReLU())  # 翘曲量为正值
        
        self.warpage_head = nn.Sequential(*warpage_modules)
        
        # 空洞预测分支
        void_modules = []
        prev_dim = shared_layers[-1]
        
        for hidden_dim in task_layers:
            void_modules.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        void_modules.append(nn.Linear(prev_dim, 1))
        void_modules.append(nn.Sigmoid())  # 空洞率在0-1之间
        
        self.void_head = nn.Sequential(*void_modules)
    
    def forward(self, x):
        """前向传播"""
        # 共享特征提取
        shared_features = self.shared_encoder(x)
        
        # 分别预测翘曲和空洞
        warpage = self.warpage_head(shared_features)
        void_ratio = self.void_head(shared_features)
        
        return torch.cat([warpage, void_ratio], dim=1)

class DNNTrainer:
    """DNN模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 learning_rate: float = 0.001):
        
        self.model = model.to(device)
        self.device = device
        
        # 损失函数 - 加权MSE
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
    
    def weighted_mse_loss(self, predictions, targets, weights=None):
        """加权MSE损失 - 对翘曲和空洞给予不同权重"""
        if weights is None:
            weights = torch.tensor([1.0, 2.0]).to(self.device)  # 空洞损失权重更高
        
        mse = (predictions - targets) ** 2
        weighted_mse = mse * weights.unsqueeze(0)
        
        return weighted_mse.mean()
    
    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            predictions = self.model(inputs)
            
            # 计算损失
            loss = self.weighted_mse_loss(predictions, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader) -> Tuple[float, Dict]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(inputs)
                loss = self.weighted_mse_loss(predictions, targets)
                
                total_loss += loss.item()
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 计算评估指标
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self._calculate_metrics(predictions, targets)
        
        return total_loss / len(dataloader), metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """计算评估指标"""
        metrics = {}
        
        # 整体指标
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['r2'] = r2_score(targets, predictions)
        
        # 分别计算翘曲和空洞指标
        warpage_pred, void_pred = predictions[:, 0], predictions[:, 1]
        warpage_true, void_true = targets[:, 0], targets[:, 1]
        
        metrics['warpage_mse'] = mean_squared_error(warpage_true, warpage_pred)
        metrics['warpage_mae'] = mean_absolute_error(warpage_true, warpage_pred)
        metrics['warpage_r2'] = r2_score(warpage_true, warpage_pred)
        
        metrics['void_mse'] = mean_squared_error(void_true, void_pred)
        metrics['void_mae'] = mean_absolute_error(void_true, void_pred)
        metrics['void_r2'] = r2_score(void_true, void_pred)
        
        return metrics
    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs: int = 100,
              early_stopping_patience: int = 15,
              save_path: str = 'models/best_dnn_model.pth'):
        """完整训练流程"""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("开始训练DNN模型...")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_metrics = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                print(f"Warpage R²: {val_metrics['warpage_r2']:.4f}, "
                      f"Void R²: {val_metrics['void_r2']:.4f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                print("-" * 50)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, save_path)
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """模型预测"""
        self.model.eval()
        
        with torch.no_grad():
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            predictions = self.model(inputs_tensor)
            
        return predictions.cpu().numpy()
    
    def evaluate_on_test_set(self, test_loader) -> Dict:
        """在测试集上评估"""
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                predictions = self.model(inputs)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # 计算详细指标
        metrics = self._detailed_metrics(predictions, targets)
        
        return metrics, predictions, targets
    
    def _detailed_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """详细评估指标"""
        metrics = {}
        
        # 整体指标
        metrics['overall'] = {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions)
        }
        
        # 翘曲指标
        warpage_pred, warpage_true = predictions[:, 0], targets[:, 0]
        metrics['warpage'] = {
            'mse': mean_squared_error(warpage_true, warpage_pred),
            'mae': mean_absolute_error(warpage_true, warpage_pred),
            'r2': r2_score(warpage_true, warpage_pred),
            'mape': np.mean(np.abs((warpage_true - warpage_pred) / warpage_true)) * 100
        }
        
        # 空洞指标
        void_pred, void_true = predictions[:, 1], targets[:, 1]
        metrics['void'] = {
            'mse': mean_squared_error(void_true, void_pred),
            'mae': mean_absolute_error(void_true, void_pred),
            'r2': r2_score(void_true, void_pred),
            'mape': np.mean(np.abs((void_true - void_pred) / void_true)) * 100
        }
        
        return metrics

if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = MoldingFlowDNN(
        input_dim=10,
        hidden_layers=[256, 128, 64, 32],
        output_dim=2,
        use_residual=True,
        use_attention=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 32
    input_dim = 10
    
    x = torch.randn(batch_size, input_dim)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Sample predictions:")
        print(f"  Warpage: {output[0, 0].item():.4f}")
        print(f"  Void ratio: {output[0, 1].item():.4f}")
    
    # 测试多任务模型
    print("\n测试多任务模型:")
    multitask_model = MultiTaskDNN(
        input_dim=10,
        shared_layers=[256, 128],
        task_layers=[64, 32]
    )
    
    with torch.no_grad():
        output = multitask_model(x)
        print(f"MultiTask output shape: {output.shape}")