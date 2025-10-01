"""
图神经网络模型 - 用于处理3D几何结构和预测翘曲
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
from loguru import logger

class GraphSAGEModel(nn.Module):
    """GraphSAGE模型用于翘曲预测"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE层
        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGE(input_dim, hidden_dim, num_layers=2))
        
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=2))
        
        # 批归一化
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 全局池化后的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        # 图卷积层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 全局池化
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_global = torch.cat([x_mean, x_max], dim=1)
        
        # 最终预测
        output = self.classifier(x_global)
        return output

class GATModel(nn.Module):
    """图注意力网络模型"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT层
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_global = torch.cat([x_mean, x_max], dim=1)
        
        output = self.classifier(x_global)
        return output

class MultiTaskGNN(nn.Module):
    """多任务GNN模型 - 同时预测翘曲和空洞"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 process_feature_dim: int = 6,
                 material_feature_dim: int = 5):
        super(MultiTaskGNN, self).__init__()
        
        # 图编码器
        self.graph_encoder = GraphSAGEModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=hidden_dim
        )
        
        # 制程和材料特征编码器
        self.process_encoder = nn.Sequential(
            nn.Linear(process_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.material_encoder = nn.Sequential(
            nn.Linear(material_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 特征融合
        fusion_dim = hidden_dim * 2 + hidden_dim // 2 + hidden_dim // 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 任务特定的头部
        self.warpage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 翘曲量预测
        )
        
        self.void_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 空洞风险预测
        )
        
    def forward(self, graph_data, process_features, material_features):
        # 提取图特征
        graph_features = self.graph_encoder(
            graph_data.x, 
            graph_data.edge_index, 
            graph_data.batch
        )
        
        # 编码制程和材料特征
        process_encoded = self.process_encoder(process_features)
        material_encoded = self.material_encoder(material_features)
        
        # 特征融合
        fused_features = torch.cat([
            graph_features, 
            process_encoded, 
            material_encoded
        ], dim=1)
        
        fused_features = self.fusion_layer(fused_features)
        
        # 多任务预测
        warpage_pred = self.warpage_head(fused_features)
        void_pred = self.void_head(fused_features)
        
        return {
            'warpage': warpage_pred,
            'void': void_pred,
            'features': fused_features
        }

class GNNTrainer:
    """GNN模型训练器"""
    
    def __init__(self, model, config_path: str = "config/config.yaml"):
        self.model = model
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['gnn']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.warpage_criterion = nn.MSELoss()
        self.void_criterion = nn.BCEWithLogitsLoss()
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        warpage_loss_sum = 0
        void_loss_sum = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(
                batch.graph_data,
                batch.process_features,
                batch.material_features
            )
            
            # 计算损失
            warpage_loss = self.warpage_criterion(
                outputs['warpage'].squeeze(), 
                batch.warpage_target
            )
            
            void_loss = self.void_criterion(
                outputs['void'].squeeze(), 
                batch.void_target
            )
            
            # 多任务损失加权
            total_batch_loss = 0.6 * warpage_loss + 0.4 * void_loss
            
            # 反向传播
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            warpage_loss_sum += warpage_loss.item()
            void_loss_sum += void_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_warpage_loss = warpage_loss_sum / len(train_loader)
        avg_void_loss = void_loss_sum / len(train_loader)
        
        return avg_loss, avg_warpage_loss, avg_void_loss
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        warpage_loss_sum = 0
        void_loss_sum = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                outputs = self.model(
                    batch.graph_data,
                    batch.process_features,
                    batch.material_features
                )
                
                warpage_loss = self.warpage_criterion(
                    outputs['warpage'].squeeze(), 
                    batch.warpage_target
                )
                
                void_loss = self.void_criterion(
                    outputs['void'].squeeze(), 
                    batch.void_target
                )
                
                total_batch_loss = 0.6 * warpage_loss + 0.4 * void_loss
                
                total_loss += total_batch_loss.item()
                warpage_loss_sum += warpage_loss.item()
                void_loss_sum += void_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_warpage_loss = warpage_loss_sum / len(val_loader)
        avg_void_loss = void_loss_sum / len(val_loader)
        
        return avg_loss, avg_warpage_loss, avg_void_loss
    
    def train(self, train_loader, val_loader, epochs: int = None):
        """完整训练流程"""
        if epochs is None:
            epochs = self.config['epochs']
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_warpage_loss, train_void_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_warpage_loss, val_void_loss = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'models/best_gnn_model.pth')
            else:
                patience_counter += 1
            
            # 日志记录
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f} (Warpage: {train_warpage_loss:.4f}, Void: {train_void_loss:.4f})")
            logger.info(f"Val Loss: {val_loss:.4f} (Warpage: {val_warpage_loss:.4f}, Void: {val_void_loss:.4f})")
            logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停
            if patience_counter >= 20:
                logger.info("早停触发")
                break
        
        logger.info("训练完成")
        return best_val_loss

if __name__ == "__main__":
    # 示例用法
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    model = MultiTaskGNN(
        input_dim=16,  # 节点特征维度
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
        process_feature_dim=6,
        material_feature_dim=5
    )
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    logger.info("GNN模型初始化完成")