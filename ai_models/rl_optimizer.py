"""
强化学习优化器
使用强化学习自动优化制程参数，最小化翘曲和空洞风险
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)

class MoldingProcessEnv(gym.Env):
    """
    模压制程环境
    强化学习智能体在此环境中学习优化制程参数
    """
    
    def __init__(self, predictor, target_warpage: float = 10.0, target_void_risk: float = 0.05):
        super(MoldingProcessEnv, self).__init__()
        
        self.predictor = predictor
        self.target_warpage = target_warpage
        self.target_void_risk = target_void_risk
        
        # 定义动作空间（制程参数的调整范围）
        self.action_space = spaces.Box(
            low=np.array([150, 5.0, 60, 20, 10e-6, 0.5, 0.8, 1.0]),    # 最小值
            high=np.array([200, 12.0, 180, 100, 25e-6, 1.2, 2.0, 3.0]), # 最大值
            dtype=np.float32
        )
        
        # 定义观察空间（当前状态 + 设计参数）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),  # 8个制程参数 + 12个设计参数
            dtype=np.float32
        )
        
        # 当前状态
        self.current_state = None
        self.design_params = None
        self.step_count = 0
        self.max_steps = 50
        
        # 历史记录
        self.history = []
        
    def reset(self, design_params: Optional[Dict] = None):
        """重置环境"""
        self.step_count = 0
        self.history = []
        
        # 设置设计参数
        if design_params is None:
            self.design_params = {
                'package_type': 'BGA',
                'chip_size': [10, 10, 0.5],
                'substrate_size': [15, 15, 0.2],
                'ball_count': 256,
                'ball_pitch': 0.8,
                'substrate_layers': 4,
                'via_density': 0.3
            }
        else:
            self.design_params = design_params
        
        # 初始化制程参数（随机初始化）
        initial_process = np.array([
            np.random.uniform(160, 190),    # 温度
            np.random.uniform(6, 10),       # 压力
            np.random.uniform(90, 150),     # 时间
            np.random.uniform(30, 70),      # 黏度
            np.random.uniform(12e-6, 20e-6), # CTE
            np.random.uniform(0.6, 1.0),    # 固化速率
            np.random.uniform(1.0, 1.8),    # 流动速率
            np.random.uniform(1.5, 2.5)     # 冷却速率
        ])
        
        # 构建完整状态
        design_features = self._extract_design_features()
        self.current_state = np.concatenate([initial_process, design_features])
        
        return self.current_state.astype(np.float32)
    
    def step(self, action):
        """执行一步动作"""
        self.step_count += 1
        
        # 更新制程参数
        process_params = {
            'temperature': float(action[0]),
            'pressure': float(action[1]),
            'time': float(action[2]),
            'viscosity': float(action[3]),
            'cte': float(action[4]),
            'cure_rate': float(action[5]),
            'flow_rate': float(action[6]),
            'cooling_rate': float(action[7])
        }
        
        # 预测性能
        try:
            prediction = self.predictor.quick_predict(process_params, self.design_params)
            warpage = prediction['max_warpage']
            void_risk = prediction['void_risk']
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            warpage = 100.0  # 惩罚值
            void_risk = 1.0
        
        # 计算奖励
        reward = self._calculate_reward(warpage, void_risk)
        
        # 更新状态
        design_features = self._extract_design_features()
        self.current_state = np.concatenate([action, design_features])
        
        # 记录历史
        self.history.append({
            'step': self.step_count,
            'action': action.tolist(),
            'warpage': warpage,
            'void_risk': void_risk,
            'reward': reward
        })
        
        # 检查终止条件
        done = (self.step_count >= self.max_steps or 
                (warpage <= self.target_warpage and void_risk <= self.target_void_risk))
        
        info = {
            'warpage': warpage,
            'void_risk': void_risk,
            'target_achieved': warpage <= self.target_warpage and void_risk <= self.target_void_risk
        }
        
        return self.current_state.astype(np.float32), reward, done, info
    
    def _extract_design_features(self) -> np.ndarray:
        """提取设计特征"""
        chip_size = self.design_params.get('chip_size', [10, 10, 0.5])
        substrate_size = self.design_params.get('substrate_size', [15, 15, 0.2])
        
        features = np.array([
            chip_size[0], chip_size[1], chip_size[2],
            substrate_size[0], substrate_size[1], substrate_size[2],
            chip_size[0] / substrate_size[0],
            (chip_size[0] * chip_size[1]) / (substrate_size[0] * substrate_size[1]),
            self.design_params.get('ball_count', 256),
            self.design_params.get('ball_pitch', 0.8),
            self.design_params.get('substrate_layers', 4),
            self.design_params.get('via_density', 0.3)
        ])
        
        return features
    
    def _calculate_reward(self, warpage: float, void_risk: float) -> float:
        """计算奖励函数"""
        # 翘曲奖励（目标：< 10μm）
        if warpage <= self.target_warpage:
            warpage_reward = 10.0 - warpage / self.target_warpage * 5.0
        else:
            warpage_reward = -5.0 * (warpage / self.target_warpage - 1.0)
        
        # 空洞风险奖励（目标：< 5%）
        if void_risk <= self.target_void_risk:
            void_reward = 10.0 - void_risk / self.target_void_risk * 5.0
        else:
            void_reward = -5.0 * (void_risk / self.target_void_risk - 1.0)
        
        # 组合奖励
        total_reward = 0.6 * warpage_reward + 0.4 * void_reward
        
        # 额外奖励：同时达到两个目标
        if warpage <= self.target_warpage and void_risk <= self.target_void_risk:
            total_reward += 20.0
        
        return float(total_reward)

class RLOptimizer:
    """
    强化学习优化器主类
    管理RL智能体的训练和推理
    """
    
    def __init__(self, model_path: str = "models/rl_checkpoints"):
        self.model_path = model_path
        self.predictor = None
        self.agent = None
        self.env = None
        self.is_loaded = False
        
        # 训练配置
        self.config = {
            'algorithm': 'PPO',  # 或 'SAC'
            'total_timesteps': 100000,
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95
        }
        
        # 训练统计
        self.training_stats = {
            'episodes': 0,
            'success_rate': 0.0,
            'best_reward': -np.inf,
            'training_history': []
        }
        
        logger.info("强化学习优化器初始化完成")
    
    def load_model(self, predictor=None):
        """加载RL模型"""
        try:
            if predictor is not None:
                self.predictor = predictor
            
            # 创建环境
            self.env = MoldingProcessEnv(self.predictor)
            
            # 加载预训练的RL智能体
            model_file = os.path.join(self.model_path, f"{self.config['algorithm'].lower()}_model.zip")
            
            if os.path.exists(model_file):
                if self.config['algorithm'] == 'PPO':
                    self.agent = PPO.load(model_file, env=self.env)
                elif self.config['algorithm'] == 'SAC':
                    self.agent = SAC.load(model_file, env=self.env)
                logger.info(f"加载预训练{self.config['algorithm']}模型成功")
            else:
                # 创建新的智能体
                self._create_new_agent()
                logger.warning(f"未找到预训练模型，创建新的{self.config['algorithm']}智能体")
            
            # 加载训练统计
            stats_file = os.path.join(self.model_path, "training_stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self.training_stats = json.load(f)
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"RL模型加载失败: {str(e)}")
            return False
    
    def _create_new_agent(self):
        """创建新的RL智能体"""
        if self.config['algorithm'] == 'PPO':
            self.agent = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=self.config['learning_rate'],
                n_steps=2048,
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                gamma=self.config['gamma'],
                gae_lambda=self.config['gae_lambda'],
                verbose=1
            )
        elif self.config['algorithm'] == 'SAC':
            self.agent = SAC(
                'MlpPolicy',
                self.env,
                learning_rate=self.config['learning_rate'],
                batch_size=self.config['batch_size'],
                gamma=self.config['gamma'],
                verbose=1
            )
    
    def optimize(self, optimization_config: Dict, design_constraints: Dict) -> Dict[str, Any]:
        """
        执行参数优化
        
        Args:
            optimization_config: 优化配置
            design_constraints: 设计约束
            
        Returns:
            优化结果
        """
        if not self.is_loaded:
            raise RuntimeError("RL模型未加载")
        
        try:
            # 设置优化目标
            self.env.target_warpage = optimization_config.get('target_warpage', 10.0)
            self.env.target_void_risk = optimization_config.get('target_void_risk', 0.05)
            max_iterations = optimization_config.get('max_iterations', 100)
            
            # 重置环境
            obs = self.env.reset(design_constraints)
            
            best_params = None
            best_performance = {'warpage': np.inf, 'void_risk': np.inf}
            optimization_history = []
            
            # 优化循环
            for iteration in range(max_iterations):
                # 智能体决策
                action, _ = self.agent.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, done, info = self.env.step(action)
                
                # 记录结果
                current_performance = {
                    'warpage': info['warpage'],
                    'void_risk': info['void_risk']
                }
                
                optimization_history.append({
                    'iteration': iteration,
                    'params': action.tolist(),
                    'performance': current_performance,
                    'reward': reward
                })
                
                # 更新最佳结果
                if (current_performance['warpage'] < best_performance['warpage'] and 
                    current_performance['void_risk'] < best_performance['void_risk']):
                    best_params = action.copy()
                    best_performance = current_performance.copy()
                
                # 检查收敛
                if done and info.get('target_achieved', False):
                    logger.info(f"优化收敛，迭代次数: {iteration + 1}")
                    break
            
            # 构建结果
            result = {
                'best_params': {
                    'temperature': float(best_params[0]),
                    'pressure': float(best_params[1]),
                    'time': float(best_params[2]),
                    'viscosity': float(best_params[3]),
                    'cte': float(best_params[4]),
                    'cure_rate': float(best_params[5]),
                    'flow_rate': float(best_params[6]),
                    'cooling_rate': float(best_params[7])
                },
                'predicted_warpage': best_performance['warpage'],
                'predicted_void_risk': best_performance['void_risk'],
                'history': optimization_history,
                'convergence': {
                    'converged': done and info.get('target_achieved', False),
                    'iterations': len(optimization_history),
                    'final_reward': optimization_history[-1]['reward'] if optimization_history else 0
                }
            }
            
            logger.info(f"参数优化完成 - 翘曲: {best_performance['warpage']:.2f}μm, "
                       f"空洞风险: {best_performance['void_risk']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"参数优化失败: {str(e)}")
            raise
    
    def generate_recipe(self, product_info: Dict) -> Dict[str, Any]:
        """
        生成智能制程卡
        
        Args:
            product_info: 产品信息
            
        Returns:
            制程卡
        """
        try:
            # 基于产品信息设置优化目标
            quality_req = product_info.get('quality_requirements', {})
            target_warpage = quality_req.get('max_warpage', 10.0)
            target_void_risk = quality_req.get('max_void_risk', 0.05)
            
            # 设计约束
            design_constraints = {
                'package_type': product_info.get('package_type', 'BGA'),
                'chip_size': product_info.get('chip_specs', {}).get('size', [10, 10, 0.5]),
                'substrate_size': product_info.get('chip_specs', {}).get('substrate_size', [15, 15, 0.2])
            }
            
            # 执行优化
            optimization_config = {
                'target_warpage': target_warpage,
                'target_void_risk': target_void_risk,
                'max_iterations': 50
            }
            
            optimization_result = self.optimize(optimization_config, design_constraints)
            
            # 生成制程卡
            recipe = {
                'product_name': product_info.get('product_name', ''),
                'recipe_id': f"RECIPE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'parameters': optimization_result['best_params'],
                'predicted_performance': {
                    'warpage': optimization_result['predicted_warpage'],
                    'void_risk': optimization_result['predicted_void_risk']
                },
                'confidence': self._calculate_recipe_confidence(optimization_result),
                'alternatives': self._generate_alternative_recipes(optimization_result),
                'created_at': datetime.now().isoformat(),
                'quality_targets': {
                    'max_warpage': target_warpage,
                    'max_void_risk': target_void_risk
                }
            }
            
            return recipe
            
        except Exception as e:
            logger.error(f"制程卡生成失败: {str(e)}")
            raise
    
    def _calculate_recipe_confidence(self, optimization_result: Dict) -> float:
        """计算制程卡置信度"""
        # 基于优化收敛性和性能指标
        convergence_score = 1.0 if optimization_result['convergence']['converged'] else 0.5
        
        # 性能得分
        warpage_score = min(1.0, 10.0 / max(optimization_result['predicted_warpage'], 1.0))
        void_score = min(1.0, 0.05 / max(optimization_result['predicted_void_risk'], 0.001))
        
        confidence = 0.4 * convergence_score + 0.3 * warpage_score + 0.3 * void_score
        return float(confidence)
    
    def _generate_alternative_recipes(self, optimization_result: Dict) -> List[Dict]:
        """生成备选制程卡"""
        # 从优化历史中选择次优解
        history = optimization_result['history']
        
        # 按奖励排序
        sorted_history = sorted(history, key=lambda x: x['reward'], reverse=True)
        
        alternatives = []
        for i, record in enumerate(sorted_history[1:4]):  # 取前3个备选
            alt_params = record['params']
            alternative = {
                'rank': i + 2,
                'parameters': {
                    'temperature': alt_params[0],
                    'pressure': alt_params[1],
                    'time': alt_params[2],
                    'viscosity': alt_params[3],
                    'cte': alt_params[4],
                    'cure_rate': alt_params[5],
                    'flow_rate': alt_params[6],
                    'cooling_rate': alt_params[7]
                },
                'predicted_performance': record['performance'],
                'reward': record['reward']
            }
            alternatives.append(alternative)
        
        return alternatives
    
    def train(self, total_timesteps: Optional[int] = None):
        """训练RL智能体"""
        if not self.is_loaded:
            raise RuntimeError("RL模型未加载")
        
        timesteps = total_timesteps or self.config['total_timesteps']
        
        # 创建训练回调
        callback = TrainingCallback(self)
        
        logger.info(f"开始训练{self.config['algorithm']}智能体，总步数: {timesteps}")
        
        # 执行训练
        self.agent.learn(
            total_timesteps=timesteps,
            callback=callback
        )
        
        # 保存模型
        self.save_model()
        
        logger.info("RL智能体训练完成")
    
    def save_model(self):
        """保存RL模型"""
        os.makedirs(self.model_path, exist_ok=True)
        
        # 保存智能体
        model_file = os.path.join(self.model_path, f"{self.config['algorithm'].lower()}_model.zip")
        self.agent.save(model_file)
        
        # 保存训练统计
        stats_file = os.path.join(self.model_path, "training_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        logger.info(f"RL模型保存至: {self.model_path}")
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.is_loaded
    
    def get_training_info(self) -> Dict:
        """获取训练信息"""
        return {
            'episodes': self.training_stats['episodes'],
            'algorithm': self.config['algorithm'],
            'total_timesteps': self.config['total_timesteps']
        }
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        return self.training_stats['success_rate']

class TrainingCallback(BaseCallback):
    """训练回调函数"""
    
    def __init__(self, optimizer):
        super(TrainingCallback, self).__init__()
        self.optimizer = optimizer
        self.episode_rewards = []
        self.episode_successes = []
    
    def _on_step(self) -> bool:
        # 记录训练进度
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                
                # 检查是否成功
                success = info.get('target_achieved', False)
                self.episode_successes.append(success)
                
                # 更新统计
                self.optimizer.training_stats['episodes'] += 1
                if len(self.episode_successes) >= 100:
                    recent_success_rate = np.mean(self.episode_successes[-100:])
                    self.optimizer.training_stats['success_rate'] = recent_success_rate
                
                # 记录最佳奖励
                if episode_reward > self.optimizer.training_stats['best_reward']:
                    self.optimizer.training_stats['best_reward'] = episode_reward
        
        return True