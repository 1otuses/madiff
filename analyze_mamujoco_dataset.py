import os
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf

def analyze_mamujoco_dataset(env_name):
    """
    分析指定MA-Mujoco环境离线数据集的规模和内容
    
    参数:
        env_name: 环境名称 (2ant, 4ant, 2halfcheetah)
    
    数据集结构：
    {env_name}/
    ├── Good/           # 专家质量数据
    ├── Medium/         # 中等质量数据
    ├── Poor/           # 低质量数据
    
    每个质量级别包含多个seed目录,每个seed包含多个.tfrecord文件
    
    转化后的npy格式:
    {env_name}/
    ├── Good/
    │   ├── obs.npy      # (n_steps, n_agents, obs_dim)
    │   ├── next_obs.npy # (n_steps, n_agents, obs_dim)
    │   ├── acs.npy      # (n_steps, n_agents, act_dim)
    │   ├── rews.npy     # (n_steps, n_agents)
    │   └── dones.npy    # (n_steps, n_agents)
    """
    
    dataset_root = Path(f"/home/lotus/data/vs_code_project/madiff/diffuser/datasets/data/mamujoco/{env_name}")
    
    print("=" * 80)
    print(f"{env_name.upper()} 离线数据集分析")
    print("=" * 80)
    print()
    
    if not dataset_root.exists():
        print(f"错误: 数据集目录不存在: {dataset_root}")
        return
    
    quality_types = ["Good", "Medium", "Poor"]
    
    file_descriptions = {
        "obs": "观测值 - 智能体对环境的观测",
        "actions": "动作值 - 智能体执行的动作",
        "rewards": "奖励值 - 智能体获得的奖励",
        "discounts": "折扣因子 - 用于计算未来奖励的折扣",
        "path_lengths": "路径长度 - 每个episode的时间步数"
    }
    
    env_info = {
        "2ant": {
            "n_agents": 2,
            "obs_dim": 78,
            "act_dim": 6,
            "description": "2个蚂蚁机器人协作搬运任务"
        },
        "4ant": {
            "n_agents": 4,
            "obs_dim": 154,
            "act_dim": 12,
            "description": "4个蚂蚁机器人协作搬运任务"
        },
        "2halfcheetah": {
            "n_agents": 2,
            "obs_dim": 28,
            "act_dim": 6,
            "description": "2个半身猎豹机器人赛跑任务"
        }
    }
    
    for quality in quality_types:
        quality_path = dataset_root / quality
        if not quality_path.exists():
            print(f"警告: {quality} 目录不存在")
            continue
            
        print(f"\n{'=' * 80}")
        print(f"数据质量: {quality.upper()}")
        print(f"{'=' * 80}")
        
        seed_dirs = sorted([d for d in quality_path.iterdir() if d.is_dir() and d.name.isdigit()])
        n_seeds = len(seed_dirs)
        
        print(f"\nSeed数量: {n_seeds}")
        
        if n_seeds == 0:
            continue
        
        first_seed = seed_dirs[0]
        print(f"\n分析示例: {first_seed.name}")
        print("-" * 80)
        
        tfrecord_files = list(first_seed.glob("*.tfrecord"))
        n_episodes = len(tfrecord_files)
        print(f"\nTFRecord文件数量 (episodes): {n_episodes}")
        
        if tfrecord_files:
            print(f"\nTFRecord文件详细信息:")
            print(f"  文件命名规则: executor_<id>_sequence_log_<episode_id>.tfrecord")
            print(f"  文件格式: TensorFlow Record Format (GZIP压缩)")
            print(f"  文件大小:")
            
            total_tfrecord_size = 0
            for tf_file in sorted(tfrecord_files)[:5]:  # 显示前5个文件的大小
                file_size = tf_file.stat().st_size
                total_tfrecord_size += file_size
                print(f"    {tf_file.name}: {file_size:,} bytes ({file_size/1024:.2f} KB)")
            
            if len(tfrecord_files) > 5:
                print(f"    ... (共{n_episodes}个文件)")
            
            avg_file_size = total_tfrecord_size / min(len(tfrecord_files), 5)
            print(f"  平均文件大小: {avg_file_size:,.0f} bytes ({avg_file_size/1024:.2f} KB)")
            
            # 实际读取TFRecord文件获取数据结构
            print(f"\nTFRecord数据结构 (实际读取):")
            try:
                # 读取第一个TFRecord文件分析数据结构
                sample_tfrecord = sorted(tfrecord_files)[0]
                dataset = tf.data.TFRecordDataset(str(sample_tfrecord), compression_type="GZIP")
                
                # 尝试读取第一个样本
                for record in dataset.take(1):
                    # 由于没有解码函数，我们只能显示文件基本信息
                    print(f"  文件: {sample_tfrecord.name}")
                    print(f"  说明: TFRecord文件包含序列化的轨迹数据")
                    print(f"  需要使用OG-MARL环境的解码函数来解析具体数据结构")
                    break
            except Exception as e:
                print(f"  读取TFRecord文件失败: {e}")
                print(f"  说明: TFRecord文件需要特定的解码函数来解析数据结构")
            
            print(f"\nTFRecord数据字段 (基于transform_og_marl_dataset.py分析):")
            print(f"  - observations: 智能体观测 (n_steps, n_agents, obs_dim)")
            print(f"  - actions: 智能体动作 (n_steps, n_agents, act_dim)")
            print(f"  - rewards: 智能体奖励 (n_steps, n_agents)")
            print(f"  - discounts: 折扣因子 (n_steps, n_agents)")
            print(f"  - extras.zero_padding_mask: 零填充掩码 (n_steps,)")
            print(f"  - extras.logprobs: 动作对数概率 (如果存在) (n_steps, n_agents)")
        
        npy_files = {}
        npy_path = quality_path  # NPY文件在quality目录下，不在seed子目录中
        for file_type in ["obs", "actions", "rewards", "discounts", "path_lengths"]:
            npy_file = npy_path / f"{file_type}.npy"
            if npy_file.exists():
                npy_files[file_type] = npy_file
        
        if npy_files:
            print(f"\n已转化NPY文件 (所有seed的综合结果):")
            print(f"  说明: NPY文件是所有{n_seeds}个seed的TFRecord文件综合起来的结果")
            print(f"  数据来源: {quality_path}/1/, {quality_path}/2/, ... 等所有seed目录")
            print(f"  数据处理: 使用np.concatenate将所有episode数据连接起来")
            print(f"  数据顺序: 按episode顺序排列,每个episode包含完整的时间步数据")
            print()
            
            for file_type, npy_file in npy_files.items():
                try:
                    data = np.load(npy_file, mmap_mode='r')
                    file_size = npy_file.stat().st_size
                    print(f"  {file_type}.npy: 形状 {data.shape}, 大小 {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                    
                    if file_type == "obs":
                        n_steps, n_agents, obs_dim = data.shape
                        print(f"    说明: {n_steps:,} 时间步, {n_agents} 智能体, {obs_dim} 维观测")
                        print(f"    数据类型: {data.dtype}")
                    elif file_type == "actions":
                        n_steps, n_agents, act_dim = data.shape
                        print(f"    说明: {n_steps:,} 时间步, {n_agents} 智能体, {act_dim} 维动作")
                        print(f"    数据类型: {data.dtype}")
                        print(f"    动作范围: [{data.min():.4f}, {data.max():.4f}]")
                    elif file_type == "rewards":
                        n_steps, n_agents = data.shape
                        print(f"    说明: {n_steps:,} 时间步, {n_agents} 智能体")
                        print(f"    数据类型: {data.dtype}")
                        print(f"    奖励统计: 最小={data.min():.4f}, 最大={data.max():.4f}, 平均={data.mean():.4f}, 标准差={data.std():.4f}")
                    elif file_type == "discounts":
                        n_steps, n_agents = data.shape
                        print(f"    说明: {n_steps:,} 时间步, {n_agents} 智能体")
                        print(f"    数据类型: {data.dtype}")
                        print(f"    折扣因子统计: 最小={data.min():.4f}, 最大={data.max():.4f}, 平均={data.mean():.4f}")
                    elif file_type == "path_lengths":
                        n_episodes = data.shape[0]
                        print(f"    说明: {n_episodes} 个episode的路径长度")
                        print(f"    数据类型: {data.dtype}")
                        print(f"    路径长度统计: 最小={data.min():.0f}, 最大={data.max():.0f}, 平均={data.mean():.2f}, 标准差={data.std():.2f}")
                        print(f"    总时间步数: {data.sum():,}")
                except Exception as e:
                    print(f"  {file_type}.npy: 加载失败 ({e})")
        else:
            print(f"\n未找到NPY文件,需要运行transform_og_marl_dataset.py转化")
            print(f"  转化命令示例: python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 2ant --quality Good")
    
    print(f"\n{'=' * 80}")
    print("总体统计")
    print(f"{'=' * 80}")
    print()
    
    print(f"{'数据质量':<15} {'Seed数':<10} {'Episodes':<15} {'智能体数':<10} {'观测维度':<10} {'动作维度':<10}")
    print("-" * 80)
    
    if env_name in env_info:
        info = env_info[env_name]
        for quality in quality_types:
            quality_path = dataset_root / quality
            if quality_path.exists():
                seed_dirs = [d for d in quality_path.iterdir() if d.is_dir() and d.name.isdigit()]
                n_seeds = len(seed_dirs)
                if n_seeds > 0:
                    first_seed = seed_dirs[0]
                    tfrecord_files = list(first_seed.glob("*.tfrecord"))
                    n_episodes = len(tfrecord_files)
                    print(f"{quality:<15} {n_seeds:<10} {n_episodes:<15} {info['n_agents']:<10} {info['obs_dim']:<10} {info['act_dim']:<10}")
    
    print(f"\n{'=' * 80}")
    print("数据集说明")
    print(f"{'=' * 80}")
    print(f"""
数据集组织结构:
- mamujoco/{env_name}/Good/   : 专家演示数据，质量最高
- mamujoco/{env_name}/Medium/ : 中等质量数据
- mamujoco/{env_name}/Poor/   : 低质量数据

原始TFRecord格式:
- 每个seed目录包含多个 .tfrecord 文件
- 每个文件存储一个episode的完整轨迹
- 文件命名: executor_<id>_sequence_log_<episode_id>.tfrecord

转化后的NPY格式:
- obs.npy         : 观测序列 (n_steps, n_agents, obs_dim)
- actions.npy     : 动作序列 (n_steps, n_agents, act_dim)
- rewards.npy     : 奖励序列 (n_steps, n_agents)
- discounts.npy    : 折扣因子序列 (n_steps, n_agents)
- path_lengths.npy : 每个episode的路径长度 (n_episodes,)

数据转化：
运行 scripts/transform_og_marl_dataset.py 将TFRecord转化为NPY格式
    """)
    
    if env_name in env_info:
        info = env_info[env_name]
        print(f"\n{'=' * 80}")
        print(f"{env_name.upper()} 环境背景")
        print(f"{'=' * 80}")
        print(f"""
{info['description']}

智能体数量: {info['n_agents']}
观测维度: {info['obs_dim']}
动作维度: {info['act_dim']}

观测空间: 包含智能体位置、速度、关节角度等信息
动作空间: 连续动作空间，控制每个关节的力矩

奖励函数: 
- 前进奖励：机器人向前移动获得正奖励
- 能量惩罚：消耗能量越少奖励越高
- 存活奖励：每个时间步获得少量正奖励
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析MA-Mujoco环境离线数据集的规模和内容')
    parser.add_argument('--env', type=str, default='2ant', 
                        choices=['2ant', '4ant', '2halfcheetah'],
                        help='选择要分析的MA-Mujoco环境类型')
    args = parser.parse_args()
    
    print(f"正在分析环境: {args.env}")
    analyze_mamujoco_dataset(args.env)