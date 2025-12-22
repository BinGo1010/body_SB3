# BodyWalk-RL

基于MuJoCo的人体行走强化学习环境。该项目使用PPO算法训练人体模型完成行走任务。

## 项目结构

```
BodyWalk-RL/
│
├── resources/               # 资源文件
│   └── body_walk/          # 人体模型文件
│
├── envs/                   # 环境定义
│   └── body_walk_env.py    # 人体行走环境
│
├── agents/                 # 强化学习代理
│   └── ppo_agent.py       # PPO算法实现
│
├── configs/                # 配置文件
│   └── body_walk_config.yaml  # 环境和训练配置
│
├── scripts/                # 运行脚本
│   ├── train_body_walk.py  # 训练脚本
│   └── evaluate_body_walk.py # 评估脚本
│
├── utils/                  # 工具函数
│   ├── mujoco_utils.py     # MuJoCo相关工具
│   └── reward_functions.py # 奖励函数
│
└── logs/                   # 训练日志
    └── body_walk/         # 模型保存路径
```

## 依赖安装

```bash
pip install numpy torch gymnasium mujoco pyyaml
```

## 使用说明

1. 训练模型：

```bash
python scripts/train_body_walk.py
```

2. 评估模型：

```bash
python scripts/evaluate_body_walk.py --model-path logs/body_walk/model_final.pt
```

## 环境配置

可以在 `configs/body_walk_config.yaml` 中修改以下配置：

- 环境参数
- 训练超参数
- 奖励系数
- 日志设置

## 奖励函数

奖励函数由以下几部分组成：

1. 前进速度奖励 ($v_{forward}$)
2. 控制能耗惩罚 ($E_{ctrl}$)
3. 横向偏移惩罚 ($p_{drift}$)
4. 跌倒惩罚 ($p_{fall}$)

总奖励计算公式：
```
reward = 1.0 * forward_vel - 0.001 * np.square(ctrl).sum() - 0.2 * drift - 5.0 * fall
```

## 许可证

MIT License