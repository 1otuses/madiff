import os
import argparse
import numpy as np
from pathlib import Path

def analyze_mpe_dataset(env_name):
    """
    分析指定MPE环境离线数据集的规模和内容
    
    参数:
        env_name: 环境名称 (simple_spread, simple_world, simple_tag)
    
    数据集结构：
    {env_name}/
    ├── expert/           # 专家质量数据
    ├── medium/           # 中等质量数据
    ├── medium-replay/    # 中等回放数据
    └── random/           # 随机数据
    
    每个质量级别包含多个seed目录,每个seed包含3个智能体的数据
    """
    
    dataset_root = Path(f"/home/lotus/data/vs_code_project/madiff/diffuser/datasets/data/mpe/{env_name}")
    
    print("=" * 80)
    print(f"{env_name.upper()} 离线数据集分析")
    print("=" * 80)
    print()
    
    # 定义数据质量类型
    quality_types = ["expert", "medium", "medium-replay", "random"]
    
    # 文件类型说明
    file_descriptions = {
        "obs": "观测值 - 智能体对环境的观测",
        "next_obs": "下一观测值 - 下一时刻的观测",
        "acs": "动作值 - 智能体执行的动作",
        "rews": "奖励值 - 智能体获得的奖励",
        "dones": "完成标志 - 标记episode是否结束"
    }
    
    total_stats = {
        "expert": {"seeds": 0, "total_steps": 0, "n_agents": 0, "obs_dim": 0, "act_dim": 0},
        "medium": {"seeds": 0, "total_steps": 0, "n_agents": 0, "obs_dim": 0, "act_dim": 0},
        "medium-replay": {"seeds": 0, "total_steps": 0, "n_agents": 0, "obs_dim": 0, "act_dim": 0},
        "random": {"seeds": 0, "total_steps": 0, "n_agents": 0, "obs_dim": 0, "act_dim": 0}
    }
    
    for quality in quality_types:
        quality_path = dataset_root / quality
        if not quality_path.exists():
            print(f"警告: {quality} 目录不存在")
            continue
            
        print(f"\n{'=' * 80}")
        print(f"数据质量: {quality.upper()}")
        print(f"{'=' * 80}")
        
        # 获取所有seed目录
        seed_dirs = sorted([d for d in quality_path.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        n_seeds = len(seed_dirs)
        
        print(f"\nSeed数量: {n_seeds}")
        
        if n_seeds == 0:
            continue
            
        # 分析第一个seed来获取数据维度信息
        first_seed = seed_dirs[0]
        print(f"\n分析示例: {first_seed.name}")
        print("-" * 80)
        
        # 获取智能体数量
        agent_files = sorted([f for f in first_seed.glob("obs_*.npy")])
        n_agents = len(agent_files)
        total_stats[quality]["n_agents"] = n_agents
        
        print(f"\n智能体数量: {n_agents}")
        
        # 分析每个智能体的数据
        for agent_idx in range(n_agents):
            print(f"\n智能体 {agent_idx} 数据:")
            print("-" * 40)
            
            obs_file = first_seed / f"obs_{agent_idx}.npy"
            next_obs_file = first_seed / f"next_obs_{agent_idx}.npy"
            acs_file = first_seed / f"acs_{agent_idx}.npy"
            rews_file = first_seed / f"rews_{agent_idx}.npy"
            dones_file = first_seed / f"dones_{agent_idx}.npy"
            
            # 加载数据获取形状信息
            if obs_file.exists():
                obs_data = np.load(obs_file, mmap_mode='r')
                n_steps, obs_dim = obs_data.shape
                total_stats[quality]["obs_dim"] = obs_dim
                total_stats[quality]["total_steps"] = n_steps * n_seeds
                total_stats[quality]["seeds"] = n_seeds
                
                print(f"  obs_{agent_idx}.npy: {file_descriptions['obs']}")
                print(f"    形状: ({n_steps:,}, {obs_dim})")
                print(f"    说明: {n_steps:,} 个时间步，每个时间步 {obs_dim} 维观测")
                
            if next_obs_file.exists():
                next_obs_data = np.load(next_obs_file, mmap_mode='r')
                n_steps_next, obs_dim_next = next_obs_data.shape
                print(f"  next_obs_{agent_idx}.npy: {file_descriptions['next_obs']}")
                print(f"    形状: ({n_steps_next:,}, {obs_dim_next})")
                print(f"    说明: {n_steps_next:,} 个时间步，每个时间步 {obs_dim_next} 维观测")
                
            if acs_file.exists():
                acs_data = np.load(acs_file, mmap_mode='r')
                n_steps_acs, act_dim = acs_data.shape
                total_stats[quality]["act_dim"] = act_dim
                
                print(f"  acs_{agent_idx}.npy: {file_descriptions['acs']}")
                print(f"    形状: ({n_steps_acs:,}, {act_dim})")
                print(f"    说明: {n_steps_acs:,} 个时间步，每个时间步 {act_dim} 维动作")
                
            if rews_file.exists():
                rews_data = np.load(rews_file, mmap_mode='r')
                n_steps_rews = rews_data.shape[0]
                
                print(f"  rews_{agent_idx}.npy: {file_descriptions['rews']}")
                print(f"    形状: ({n_steps_rews:,},)")
                print(f"    说明: {n_steps_rews:,} 个时间步，每个时间步 1 维奖励")
                
            if dones_file.exists():
                dones_data = np.load(dones_file, mmap_mode='r')
                n_steps_dones = dones_data.shape[0]
                
                print(f"  dones_{agent_idx}.npy: {file_descriptions['dones']}")
                print(f"    形状: ({n_steps_dones:,},)")
                print(f"    说明: {n_steps_dones:,} 个时间步，每个时间步 1 维完成标志")
    
    # 输出总体统计
    print(f"\n{'=' * 80}")
    print("总体统计")
    print(f"{'=' * 80}")
    print()
    
    print(f"{'数据质量':<15} {'Seed数':<10} {'总步数':<15} {'智能体数':<10} {'观测维度':<10} {'动作维度':<10}")
    print("-" * 80)
    
    grand_total_steps = 0
    for quality in quality_types:
        stats = total_stats[quality]
        if stats["seeds"] > 0:
            print(f"{quality:<15} {stats['seeds']:<10} {stats['total_steps']:<15,} "
                  f"{stats['n_agents']:<10} {stats['obs_dim']:<10} {stats['act_dim']:<10}")
            grand_total_steps += stats['total_steps']
    
    print("-" * 80)
    print(f"{'总计':<15} {'-':<10} {grand_total_steps:<15,}")
    
    # 数据集说明
    print(f"\n{'=' * 80}")
    print("数据集说明")
    print(f"{'=' * 80}")
    print("""
数据集组织结构：
- simple_spread/expert/       : 专家演示数据，质量最高
- simple_spread/medium/       : 中等质量数据
- simple_spread/medium-replay/: 中等回放数据

每个质量级别包含：
- 多个seed目录 (seed_0_data, seed_1_data, ...)
- 每个seed包含3个智能体的完整轨迹数据

文件命名规则：
- obs_{agent_id}.npy      : 智能体agent_id的观测序列
- next_obs_{agent_id}.npy : 智能体agent_id的下一观测序列
- acs_{agent_id}.npy      : 智能体agent_id的动作序列
- rews_{agent_id}.npy     : 智能体agent_id的奖励序列
- dones_{agent_id}.npy    : 智能体agent_id的完成标志序列

数据维度说明：
- 观测维度: 18维 (包含位置、速度、邻居信息等)
- 动作维度: 2维 (连续动作空间，x和y方向的移动)
- 智能体数量: 3个
- 每个seed包含约100万时间步的数据

数据用途：
- 离线强化学习训练
- 行为克隆 (Behavior Cloning)
- 多智能体协调学习
- 策略评估和比较
    """)
    
    print(f"\n{'=' * 80}")
    print("文件内容详细说明")
    print(f"{'=' * 80}")
    print()
    
    for file_key, description in file_descriptions.items():
        print(f"{file_key:<12} : {description}")
    
    print(f"\n{'=' * 80}")
    print("Simple Spread 环境背景")
    print(f"{'=' * 80}")
    print("""
Simple Spread 是一个多智能体粒子环境任务：
- 3个智能体需要协作将3个 landmarks（地标）散开
- 每个智能体被分配到一个特定的landmark
- 智能体需要通过移动来覆盖其分配的landmark
- 当所有landmark都被覆盖且智能体之间保持一定距离时获得奖励
- 需要智能体之间的协调和避免碰撞

观测空间 (18维):
- 智能体自身的位置和速度信息
- 其他智能体的相对位置和速度信息
- 目标landmark的位置信息
- 环境边界信息等

动作空间 (2维):
- 连续动作：[x方向移动, y方向移动]
- 动作范围通常在[-1, 1]之间

奖励函数:
- 覆盖奖励：智能体接近其目标landmark时获得奖励
- 距离惩罚：智能体之间距离太近时受到惩罚
- 碰撞惩罚：智能体之间发生碰撞时受到惩罚
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析MPE环境离线数据集的规模和内容')
    parser.add_argument('--env', type=str, default='simple_spread', 
                        choices=['simple_spread', 'simple_world', 'simple_tag'],
                        help='选择要分析的MPE环境类型')
    args = parser.parse_args()
    
    print(f"正在分析环境: {args.env}")
    analyze_mpe_dataset(args.env)