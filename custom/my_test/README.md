# my_test

Risk-guided diffusion prototype on MPE (simple_spread/simple_tag/simple_world). This folder is self-contained and does not use ml_logger.

## Environment

Activate conda env in every new terminal:

```bash
conda activate madiff
```

Optional (avoid GR prompt and enable headless render):

```bash
export SUPPRESS_GR_PROMPT=1
export SDL_VIDEODRIVER=dummy
```

## Paths

- Config: `custom/my_test/config/mpe_spread_exp.yaml` (see other env/quality configs in `custom/my_test/config/`)
- Checkpoints: `runs/my_test/{env_name}/{quality}/checkpoint/checkpoint.pt`
- TensorBoard: `runs/my_test/{env_name}/{quality}/tensorboard`
- Videos: `runs/my_test/{env_name}/{quality}/videos`

## Training (step-based)

```bash
conda activate madiff
python custom/my_test/run_scripts/train.py \
  -c custom/my_test/config/mpe_spread_exp.yaml \
  --device cuda
```

Override steps for quick smoke test:

```bash
conda activate madiff
python custom/my_test/run_scripts/train.py \
  -c custom/my_test/config/mpe_spread_exp.yaml \
  --device cpu \
  --n_train_steps 200
```

## Evaluation (online, with video)

```bash
conda activate madiff
SUPPRESS_GR_PROMPT=1 SDL_VIDEODRIVER=dummy \
python custom/my_test/run_scripts/evaluate.py \
  -c custom/my_test/config/mpe_spread_exp.yaml \
  --checkpoint runs/my_test/simple_spread/expert/checkpoint/checkpoint.pt \
  --device cuda
```

Skip video:

```bash
conda activate madiff
SUPPRESS_GR_PROMPT=1 \
python custom/my_test/run_scripts/evaluate.py \
  -c custom/my_test/config/mpe_spread_exp.yaml \
  --checkpoint runs/my_test/simple_spread/expert/checkpoint/checkpoint.pt \
  --device cuda \
  --no_video
```

## End-to-end (train + eval)

```bash
conda activate madiff
SUPPRESS_GR_PROMPT=1 SDL_VIDEODRIVER=dummy \
python custom/my_test/run_experiment.py \
  -c custom/my_test/config/mpe_spread_exp.yaml \
  --device cuda
```

## Metrics

Evaluation JSON (`runs/my_test/{env_name}/eval/eval_results.json`) contains:
- `average_ep_reward`: per-agent episode return mean (MADiff-style)
- `std_ep_reward`: per-agent episode return std
- `overall_mean` / `overall_std`: sum over agents per episode (for comparison)

Training logs are written to TensorBoard; `log_freq` is adaptive for short runs.
