"""标准 evaluate.py 可动态加载的冻结 mode evaluators。"""

import copy
import json
import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from diffuser.utils.arrays import batch_to_device

from .metrics import (
    local_mode_evaluation,
    mode_discovery_metrics,
    normalized_mutual_information,
    terminal_assignment,
)


class _ModeEvaluatorBase:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.initialized = False

    def init(
        self,
        log_dir: str,
        num_eval: int = 256,
        condition_guidance_w=None,
        mode_guidance_w=None,
        use_ddim_sample: bool = False,
        n_ddim_steps: int = 15,
        local_prefix_eval: int = 1,
        local_prefixes=None,
        local_confidence_threshold: float = 0.0,
        dataset_split: str = "eval",
        **kwargs,
    ):
        del kwargs
        if num_eval < 1:
            raise ValueError("num_eval must be positive")
        if local_prefix_eval < 1:
            raise ValueError("local_prefix_eval must be positive")
        if not 0.0 <= local_confidence_threshold <= 1.0:
            raise ValueError("local_confidence_threshold must be in [0, 1]")
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.num_eval = num_eval
        self.condition_guidance_w = condition_guidance_w
        self.mode_guidance_w = mode_guidance_w
        self.use_ddim_sample = use_ddim_sample
        self.n_ddim_steps = n_ddim_steps
        self.local_prefix_eval = local_prefix_eval
        self.local_prefixes = local_prefixes
        self.local_confidence_threshold = local_confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with (self.log_dir / "dataset_config.pkl").open("rb") as stream:
            dataset_config = pickle.load(stream)
        with (self.log_dir / "model_config.pkl").open("rb") as stream:
            self.model_config = pickle.load(stream)
        with (self.log_dir / "diffusion_config.pkl").open("rb") as stream:
            self.objective_config = pickle.load(stream)
        dataset_config = copy.deepcopy(dataset_config)
        dataset_config._dict["split"] = dataset_split
        self.dataset = dataset_config()
        requested_prefixes = self.local_prefixes or [1, self.dataset.horizon]
        requested_prefixes = list(requested_prefixes) + [self.local_prefix_eval]
        self.local_prefixes = sorted(
            {
                min(max(1, int(prefix)), self.dataset.horizon)
                for prefix in requested_prefixes
            }
        )
        self.local_prefix_eval = min(self.local_prefix_eval, self.dataset.horizon)
        self.initialized = True

    def _load_objective(self, load_step: int):
        model = self.model_config().to(self.device)
        objective = self.objective_config(model).to(self.device)
        checkpoint_path = self.log_dir / "checkpoint" / "state_{}.pt".format(load_step)
        if not checkpoint_path.exists():
            checkpoint_path = self.log_dir / "checkpoint" / "state.pt"
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("ema", checkpoint.get("model", checkpoint))
        objective.load_state_dict(state_dict, strict=True)
        objective.eval()
        return objective, checkpoint_path

    def _loader(self):
        count = min(int(self.num_eval), len(self.dataset))
        subset = torch.utils.data.Subset(self.dataset, list(range(count)))
        return torch.utils.data.DataLoader(
            subset,
            batch_size=min(256, count),
            shuffle=False,
            num_workers=0,
        )

    def _save(self, load_step: int, result: Dict) -> None:
        results_dir = self.log_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        suffix = "-ddim" if self.use_ddim_sample else ""
        if self.condition_guidance_w is not None:
            suffix += "-cg_{}".format(self.condition_guidance_w)
        if self.mode_guidance_w is not None:
            suffix += "-mg_{}".format(self.mode_guidance_w)
        if getattr(self, "use_return_to_go", False):
            suffix += "-rtg"
        path = results_dir / "step_{}-ep_{}{}.json".format(
            load_step, self.num_eval, suffix
        )
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self.verbose:
            print(json.dumps(result, ensure_ascii=False, indent=2))


class ModeVQEvaluator(_ModeEvaluatorBase):
    """评估 code usage、审计标签关系和 local-prior 可辨识性。"""

    @torch.no_grad()
    def evaluate(self, load_step: int, **kwargs):
        del kwargs
        if not self.initialized:
            raise RuntimeError("evaluator.init must be called first")
        objective, checkpoint_path = self._load_objective(load_step)
        model = objective.mode_model
        posterior_codes = []
        observations = []
        actions = []
        masks = []
        for batch in self._loader():
            batch = batch_to_device(batch, self.device)
            output = model(**batch)
            posterior_codes.append(output["codes"].cpu().numpy())
            observations.append(batch["observations"])
            actions.append(batch["actions"])
            masks.append(batch["mask"])

        posterior_codes = np.concatenate(posterior_codes)
        all_observations = torch.cat(observations)
        all_actions = torch.cat(actions)
        all_masks = torch.cat(masks)
        tensor_codes = torch.as_tensor(posterior_codes, device=self.device)
        reconstruction_mse = {}
        code_variants = {
            "posterior": tensor_codes,
            "shuffle": tensor_codes.roll(1),
            "none": torch.full_like(tensor_codes, -1),
        }
        valid = all_masks[..., None, None].to(all_actions.dtype)
        count = float((valid.sum() * model.n_agents * model.action_dim).cpu())
        for name, codes in code_variants.items():
            predicted = model.decode_actions(all_observations, codes)
            squared_error = ((predicted - all_actions).square() * valid).sum()
            reconstruction_mse[name] = float(squared_error.cpu()) / max(count, 1.0)

        local_evidence = {}
        for prefix in self.local_prefixes:
            local_codes = []
            confidences = []
            for batch in self._loader():
                batch = batch_to_device(batch, self.device)
                local = model.infer_local_modes(
                    **_prefix_batch(batch, prefix),
                    confidence_threshold=self.local_confidence_threshold,
                )
                local_codes.append(local["codes"].cpu().numpy())
                confidences.append(local["confidence"].cpu().numpy())
            local_codes = np.concatenate(local_codes)
            confidences = np.concatenate(confidences)
            local_evidence[str(prefix)] = {
                **local_mode_evaluation(
                    local_codes, posterior_codes, n_modes=model.n_modes
                ),
                "mean_confidence": float(confidences.mean()),
            }
        usage = np.bincount(posterior_codes, minlength=model.n_modes)
        probabilities = usage[usage > 0] / max(usage.sum(), 1)
        result = {
            "status": "offline_evaluation_complete",
            "review_status": "pending_user_discussion",
            "checkpoint": str(checkpoint_path),
            "environment_rollout": False,
            "posterior": {
                "mode_usage": usage.tolist(),
                "active_codes": int(np.count_nonzero(usage)),
                "hard_perplexity": float(
                    np.exp(-(probabilities * np.log(probabilities)).sum())
                ),
                "reconstruction_mse": reconstruction_mse["posterior"],
                "shuffle_reconstruction_mse": reconstruction_mse["shuffle"],
                "no_mode_reconstruction_mse": reconstruction_mse["none"],
            },
            "local_prior": local_evidence[str(self.local_prefix_eval)],
            "local_prior_by_prefix": local_evidence,
        }
        indices = np.arange(len(posterior_codes))
        true_modes = self.dataset.audit_labels(indices, "true_modes")
        if true_modes is not None:
            true_mode_metrics = mode_discovery_metrics(posterior_codes, true_modes)
            result["posterior"].update(
                mode_accuracy=true_mode_metrics["mode_accuracy"],
                mode_nmi=true_mode_metrics["mode_nmi"],
                mode_ari=true_mode_metrics["mode_ari"],
                mode_mapping=true_mode_metrics["mode_mapping"],
            )
        collector_ids = self.dataset.audit_labels(indices, "collector_ids")
        if collector_ids is not None:
            result["posterior"]["collector_nmi"] = normalized_mutual_information(
                posterior_codes, collector_ids
            )
        terminal_assignments = _window_terminal_assignments(self.dataset, indices)
        if terminal_assignments is not None:
            assignment_metrics = mode_discovery_metrics(
                posterior_codes, terminal_assignments
            )
            result["posterior"].update(
                terminal_assignment_accuracy=assignment_metrics["mode_accuracy"],
                terminal_assignment_nmi=assignment_metrics["mode_nmi"],
                terminal_assignment_ari=assignment_metrics["mode_ari"],
            )
        self._save(load_step, result)
        return result


class ModeConditionedEvaluator(_ModeEvaluatorBase):
    """固定随机噪声比较 posterior/local/shuffle/no-mode 的离线动作误差。"""

    @torch.no_grad()
    def evaluate(self, load_step: int, **kwargs):
        del kwargs
        if not self.initialized:
            raise RuntimeError("evaluator.init must be called first")
        if self.local_prefix_eval != 1:
            raise ValueError(
                "first-action evaluation requires local_prefix_eval=1 to avoid "
                "target-action leakage"
            )
        diffusion, checkpoint_path = self._load_objective(load_step)
        if self.condition_guidance_w is not None:
            diffusion.condition_guidance_w = self.condition_guidance_w
        if self.mode_guidance_w is not None:
            diffusion.mode_guidance_w = self.mode_guidance_w
        if self.use_ddim_sample:
            diffusion.set_ddim_scheduler(self.n_ddim_steps)

        squared_errors = {
            "posterior": 0.0,
            "local": 0.0,
            "shuffle": 0.0,
            "none": 0.0,
        }
        element_counts = {key: 0 for key in squared_errors}
        for batch_index, batch in enumerate(self._loader()):
            batch = batch_to_device(batch, self.device)
            observations = batch["x"][..., diffusion.action_dim :]
            actions = batch["x"][..., : diffusion.action_dim]
            posterior = diffusion.infer_mode_codes(
                observations, actions, batch["mode_mask"], source="posterior"
            )
            local = diffusion.infer_mode_codes(
                observations[:, : self.local_prefix_eval],
                actions[:, : self.local_prefix_eval],
                batch["mode_mask"][:, : self.local_prefix_eval],
                source="local",
            )
            shuffle = posterior.roll(1, dims=0)
            none = torch.full(
                (actions.shape[0], diffusion.n_agents),
                -1,
                device=self.device,
                dtype=torch.long,
            )
            conditions = {
                "posterior": posterior,
                "local": local,
                "shuffle": shuffle,
                "none": none,
            }
            target = actions[:, 0]
            for name, codes in conditions.items():
                seed = 1000003 + batch_index
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                sample = diffusion.conditional_sample(
                    batch["cond"],
                    returns=batch.get("returns"),
                    attention_masks=batch.get("attention_masks"),
                    verbose=False,
                    model_kwargs={"mode_codes": codes},
                )
                if diffusion.use_inv_dyn:
                    predicted_action = diffusion.predict_actions(sample)[:, 0]
                else:
                    predicted_action = sample[:, 0, :, : diffusion.action_dim]
                error = (predicted_action - target).square()
                squared_errors[name] += float(error.sum().cpu())
                element_counts[name] += error.numel()

        result = {
            "status": "offline_evaluation_complete",
            "review_status": "pending_user_discussion",
            "checkpoint": str(checkpoint_path),
            "environment_rollout": False,
            "diffusion_target": (
                "observations" if diffusion.use_inv_dyn else "actions_observations"
            ),
            "action_decoder": (
                "local_inverse_dynamics"
                if diffusion.use_inv_dyn
                else "direct_diffusion"
            ),
            "first_action_mse": {
                key: squared_errors[key] / max(element_counts[key], 1)
                for key in squared_errors
            },
            "conditions": ["posterior", "local", "shuffle", "none"],
        }
        self._save(load_step, result)
        return result


def _prefix_batch(batch: Dict[str, torch.Tensor], prefix: int):
    return {
        "observations": batch["observations"][:, :prefix],
        "actions": batch["actions"][:, :prefix],
        "mask": batch["mask"][:, :prefix],
    }


def _window_terminal_assignments(dataset, indices):
    metadata = dataset.store.metadata or {}
    environment = metadata.get("task", metadata.get("environment"))
    if (
        environment != "simple_spread"
        or dataset.n_agents != 3
        or dataset.observation_dim < 10
    ):
        return None

    assignments = []
    for dataset_index in np.asarray(indices, dtype=np.int64):
        episode, start = dataset.indices[int(dataset_index)]
        end = min(start + dataset.horizon, dataset.store.horizon)
        valid = np.flatnonzero(dataset.store.mask[episode, start:end])
        if not len(valid):
            raise ValueError("evaluation window has no valid terminal observation")
        observation = dataset.store.observations[episode, start + int(valid[-1])]
        assignments.append(terminal_assignment(observation))
    _, labels = np.unique(np.asarray(assignments), axis=0, return_inverse=True)
    return labels
