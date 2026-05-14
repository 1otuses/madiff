#!/bin/bash
# smac maps
python scripts/transform_og_marl_dataset.py --env_name smac --map_name 3m --quality Good
python scripts/transform_og_marl_dataset.py --env_name smac --map_name 3m --quality Medium
python scripts/transform_og_marl_dataset.py --env_name smac --map_name 3m --quality Poor

python scripts/transform_og_marl_dataset.py --env_name smac --map_name 8m --quality Good
python scripts/transform_og_marl_dataset.py --env_name smac --map_name 8m --quality Medium
python scripts/transform_og_marl_dataset.py --env_name smac --map_name 8m --quality Poor

python scripts/transform_og_marl_dataset.py --env_name smac --map_name 2s3z --quality Good
python scripts/transform_og_marl_dataset.py --env_name smac --map_name 2s3z --quality Medium
python scripts/transform_og_marl_dataset.py --env_name smac --map_name 2s3z --quality Poor

python scripts/transform_og_marl_dataset.py --env_name smac --map_name 5m_vs_6m --quality Good
python scripts/transform_og_marl_dataset.py --env_name smac --map_name 5m_vs_6m --quality Medium
python scripts/transform_og_marl_dataset.py --env_name smac --map_name 5m_vs_6m --quality Poor

# mamujoco maps
python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 2ant --quality Good
python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 2ant --quality Medium
python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 2ant --quality Poor

python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 4ant --quality Good
python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 4ant --quality Medium
python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 4ant --quality Poor

python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 2halfcheetah --quality Good
python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 2halfcheetah --quality Medium
python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name 2halfcheetah --quality Poor
