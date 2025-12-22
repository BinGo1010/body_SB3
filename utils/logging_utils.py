"""工具函数：用于记录训练过程中的奖励和指标"""

import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import time

class RewardLogger:
    """奖励和指标记录器"""
    
    def __init__(self, log_dir: str):
        """初始化记录器
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前回合的奖励和指标
        self.current_rewards = {
            'forward_reward': [],
            'ctrl_cost': [],
            'drift_penalty': [],
            'fall_penalty': [],
            'total': []
        }
        
        self.current_metrics = {
            'forward_vel': [],
            'height': [],
            'drift': [],
            'ctrl_magnitude': []
        }
        
        # 所有回合的统计信息
        self.episode_stats = []
        
        # 创建日志文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.rewards_file = self.log_dir / f"rewards_{timestamp}.jsonl"
        self.summary_file = self.log_dir / f"summary_{timestamp}.json"
    
    def add_step(self, info: Dict):
        """添加一个步骤的信息
        
        Args:
            info: 环境返回的info字典
        """
        if 'rewards' not in info or 'metrics' not in info:
            return
            
        for k, v in info['rewards'].items():
            if k in self.current_rewards:
                self.current_rewards[k].append(v)
                
        for k, v in info['metrics'].items():
            if k in self.current_metrics:
                self.current_metrics[k].append(v)
    
    def end_episode(self, episode_num: int, steps: int):
        """结束当前回合，计算统计信息
        
        Args:
            episode_num: 回合编号
            steps: 回合步数
        """
        # 计算统计量
        stats = {
            'episode': episode_num,
            'steps': steps,
            'timestamp': time.time(),
            'rewards': {},
            'metrics': {}
        }
        
        # 奖励统计
        for k, v in self.current_rewards.items():
            if v:  # 如果列表非空
                stats['rewards'][k] = {
                    'mean': float(np.mean(v)),
                    'std': float(np.std(v)),
                    'min': float(np.min(v)),
                    'max': float(np.max(v)),
                    'sum': float(np.sum(v))
                }
        
        # 指标统计
        for k, v in self.current_metrics.items():
            if v:  # 如果列表非空
                stats['metrics'][k] = {
                    'mean': float(np.mean(v)),
                    'std': float(np.std(v)),
                    'min': float(np.min(v)),
                    'max': float(np.max(v))
                }
        
        # 保存统计信息
        self.episode_stats.append(stats)
        with open(self.rewards_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
        
        # 清空当前回合数据
        for k in self.current_rewards:
            self.current_rewards[k] = []
        for k in self.current_metrics:
            self.current_metrics[k] = []
        
        return stats
    
    def save_summary(self):
        """保存训练过程的总体统计信息"""
        if not self.episode_stats:
            return
            
        summary = {
            'total_episodes': len(self.episode_stats),
            'total_steps': sum(ep['steps'] for ep in self.episode_stats),
            'rewards_summary': {},
            'metrics_summary': {}
        }
        
        # 汇总所有回合的奖励
        for reward_key in self.episode_stats[0]['rewards'].keys():
            values = [ep['rewards'][reward_key]['mean'] for ep in self.episode_stats]
            summary['rewards_summary'][reward_key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # 汇总所有回合的指标
        for metric_key in self.episode_stats[0]['metrics'].keys():
            values = [ep['metrics'][metric_key]['mean'] for ep in self.episode_stats]
            summary['metrics_summary'][metric_key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary