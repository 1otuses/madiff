import argparse
import datetime
import os
from subprocess import Popen
from time import sleep

import dateutil.tz
import yaml

from diffuser.utils.launcher_util import RUN, build_nested_variant_generator

if __name__ == "__main__":
    """
    实验运行脚本
    
    功能说明：
    1. 解析命令行参数(实验配置文件和GPU ID)
    2. 加载YAML格式的实验配置规范
    3. 生成所有实验变体(基于配置中的参数组合)
    4. 创建时间戳目录保存所有变体配置
    5. 使用多进程并行运行实验(支持指定worker数量)
    
    使用方法：
        python run_experiment.py -e <config_file.yaml> -g <gpu_id>
    
    参数说明：
        -e/--exp_config: 实验配置文件的路径
        -g/--gpu: 使用的GPU ID,默认为0
    """

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_config", help="experiment config file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()

    with open(args.exp_config, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.FullLoader)

    # generating the variants
    # 使用配置生成器创建所有可能的实验变体组合
    vg_fn = build_nested_variant_generator(exp_specs)

    # 创建时间戳和日志目录
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    variants_log_dir = os.path.join(
        RUN.script_root, # ~/madiff
        f"logs/variants/variants-for-{exp_specs['meta_data']['exp_name']}",
        "variants-" + timestamp,
    )
    os.makedirs(variants_log_dir)
    
    # 保存原始实验配置定义
    with open(os.path.join(variants_log_dir, "exp_spec_definition.yaml"), "w") as f:
        yaml.dump(exp_specs, f, default_flow_style=False)
    
    # 生成并保存所有实验变体配置文件
    num_variants = 0
    for variant in vg_fn():
        i = num_variants
        variant["exp_id"] = i
        with open(os.path.join(variants_log_dir, "%d.yaml" % i), "w") as f:
            yaml.dump(variant, f, default_flow_style=False)
            f.flush()
        num_variants += 1

    # 确定worker数量（不超过变体总数）
    num_workers = min(exp_specs["meta_data"]["num_workers"], num_variants)
    exp_specs["meta_data"]["num_workers"] = num_workers

    # run the processes
    # 并行运行实验进程，维护一个运行中的进程列表
    running_processes = []
    args_idx = 0

    # 构建运行命令模板
    command = "python {script_path} -e {specs} -g {gpuid}"
    command_format_dict = exp_specs["meta_data"]

    # 主循环：启动和管理实验进程
    print(f"\n{'='*60}")
    print(f"开始运行实验: {exp_specs['meta_data']['exp_name']}")
    print(f"总变体数: {num_variants}, Worker数量: {num_workers}, GPU ID: {args.gpu}")
    print(f"{'='*60}\n")
    
    while (args_idx < num_variants) or (len(running_processes) > 0):
        # 如果还有未启动的变体且worker未满，启动新进程
        if (len(running_processes) < num_workers) and (args_idx < num_variants):
            command_format_dict["specs"] = os.path.join(
                variants_log_dir, "%i.yaml" % args_idx
            )
            command_format_dict["gpuid"] = args.gpu
            command_to_run = command.format(**command_format_dict)
            command_to_run = command_to_run.split()
            print(f"[启动进程 {args_idx+1}/{num_variants}] 命令: {' '.join(command_to_run)}")
            p = Popen(command_to_run)
            args_idx += 1
            running_processes.append(p)
            print(f"  → 进程 {args_idx} 已启动 (PID: {p.pid})")
            print(f"  → 当前运行进程数: {len(running_processes)}/{num_workers}")
        else:
            # 等待1秒后检查进程状态
            sleep(1)

        # 清理已完成的进程，保留仍在运行的进程
        previous_count = len(running_processes)
        new_running_processes = []
        for p in running_processes:
            ret_code = p.poll()
            if ret_code is None:
                new_running_processes.append(p)
            else:
                print(f"[进程完成] PID: {p.pid}, 返回码: {ret_code}")
        running_processes = new_running_processes
        
        # 如果有进程完成，显示当前状态
        if len(running_processes) != previous_count:
            completed = args_idx - len(running_processes)
            print(f"  → 进度: {completed}/{num_variants} 完成, {len(running_processes)} 运行中")
            print(f"  → 剩余待启动: {num_variants - args_idx}\n")