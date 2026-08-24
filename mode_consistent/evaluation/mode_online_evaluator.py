"""Mode-conditioned diffusion 在真实 MPE 环境中的冻结 rollout evaluator。"""

import gc
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from diffuser.datasets.mpe import load_environment
from diffuser.utils.setup import set_seed
from diffuser.utils.video import save_video

from .mode_evaluator import _ModeEvaluatorBase


class ModeOnlinePolicy:
    """用局部 mode prior、observation diffusion 和 local inverse dynamics 决策。"""

    def __init__(
        self,
        diffusion,
        normalizer,
        device: torch.device,
        target_return: Optional[float],
        returns_scale: float = 400.0,
        discount: float = 0.99,
        use_return_to_go: bool = False,
        local_confidence_threshold: float = 0.0,
    ):
        if diffusion.mode_model is None:
            raise ValueError("online mode inference requires a frozen stage1 model")
        if not diffusion.use_inv_dyn or diffusion.joint_inv:
            raise ValueError("online rollout requires local inverse dynamics")
        if diffusion.returns_condition and target_return is None:
            raise ValueError("return-conditioned diffusion requires target_return")
        if returns_scale <= 0.0:
            raise ValueError("returns_scale must be positive")
        if use_return_to_go and discount <= 0.0:
            raise ValueError("discount must be positive when updating return-to-go")
        if not 0.0 <= local_confidence_threshold <= 1.0:
            raise ValueError("local_confidence_threshold must be in [0, 1]")
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.device = device
        self.initial_target_return = (
            None if target_return is None else float(target_return)
        )
        self.target_return = self.initial_target_return
        self.returns_scale = float(returns_scale)
        self.discount = float(discount)
        self.use_return_to_go = bool(use_return_to_go)
        self.local_confidence_threshold = local_confidence_threshold

    def reset_return_to_go(self) -> None:
        self.target_return = self.initial_target_return

    def update_return_to_go(self, reward: np.ndarray) -> None:
        if not self.use_return_to_go or self.target_return is None:
            return
        reward = np.asarray(reward, dtype=np.float64)
        if reward.size == 0:
            raise ValueError("reward must not be empty")
        team_reward = float(reward.mean())
        raw_return = self.target_return * self.returns_scale
        self.target_return = (
            (raw_return - team_reward) / self.discount / self.returns_scale
        )

    @torch.no_grad()
    def act(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        observation_history = np.asarray(observation_history, dtype=np.float32)
        action_history = np.asarray(action_history, dtype=np.float32)
        expected_observation_shape = (
            self.diffusion.n_agents,
            self.diffusion.observation_dim,
        )
        expected_action_shape = (
            self.diffusion.n_agents,
            self.diffusion.action_dim,
        )
        if (
            observation_history.ndim != 3
            or observation_history.shape[1:] != expected_observation_shape
        ):
            raise ValueError(
                "observation_history must have shape [time, agent, observation]"
            )
        if action_history.shape != (
            observation_history.shape[0] - 1,
            *expected_action_shape,
        ):
            raise ValueError(
                "action_history must contain one fewer step than observation_history"
            )

        normalized_observations = self.normalizer.normalize(
            observation_history, "observations"
        ).astype(np.float32)
        aligned_actions = np.zeros(
            (observation_history.shape[0], *expected_action_shape), dtype=np.float32
        )
        if len(action_history):
            aligned_actions[:-1] = self.normalizer.normalize(
                action_history, "actions"
            ).astype(np.float32)

        observations = torch.as_tensor(
            normalized_observations[None], device=self.device
        )
        actions = torch.as_tensor(aligned_actions[None], device=self.device)
        mask = torch.ones(
            (1, observation_history.shape[0]),
            device=self.device,
            dtype=observations.dtype,
        )
        local = self.diffusion.mode_model.infer_local_modes(
            observations,
            actions,
            mask,
            confidence_threshold=self.local_confidence_threshold,
        )

        horizon = self.diffusion.horizon
        cond_x = torch.zeros(
            (
                1,
                horizon,
                self.diffusion.n_agents,
                self.diffusion.observation_dim,
            ),
            device=self.device,
            dtype=observations.dtype,
        )
        cond_masks = torch.zeros_like(cond_x, dtype=torch.bool)
        cond_x[:, 0] = observations[:, -1]
        cond_masks[:, 0] = True
        attention_masks = torch.ones(
            (1, horizon, self.diffusion.n_agents, 1),
            device=self.device,
            dtype=observations.dtype,
        )
        attention_masks[:, 0] = 0.0
        returns = None
        if self.diffusion.returns_condition:
            returns = torch.full(
                (1, 1, self.diffusion.n_agents),
                float(self.target_return),
                device=self.device,
                dtype=observations.dtype,
            )

        sampled_observations = self.diffusion.conditional_sample(
            {"x": cond_x, "masks": cond_masks},
            returns=returns,
            attention_masks=attention_masks,
            verbose=False,
            model_kwargs={"mode_codes": local["codes"]},
        )
        normalized_action = self.diffusion.predict_actions(sampled_observations)[:, 0]
        action = self.normalizer.unnormalize(
            normalized_action[0].cpu().numpy(), "actions"
        ).astype(np.float32)
        return action, {
            "mode_codes": local["codes"][0].cpu().tolist(),
            "mode_confidence": local["confidence"][0].cpu().tolist(),
        }


def rollout_mpe_episode(
    env,
    policy,
    max_episode_steps: int,
    capture_video: bool,
) -> Dict:
    """执行一个真实 MPE episode；return 只累计环境实际 reward。"""

    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    reset_return_to_go = getattr(policy, "reset_return_to_go", None)
    if callable(reset_return_to_go):
        reset_return_to_go()
    observation = np.asarray(env.reset(), dtype=np.float32)
    observation_history = [observation.copy()]
    action_history = []
    mode_codes = []
    mode_confidences = []
    frames = [_render_frame(env)] if capture_video else []
    agent_return = np.zeros(observation.shape[0], dtype=np.float64)

    for _ in range(max_episode_steps):
        actions = (
            np.stack(action_history)
            if action_history
            else _empty_action_history(env, observation.shape[0])
        )
        action, policy_info = policy.act(
            np.stack(observation_history),
            actions,
        )
        action = _clip_actions(action, env.action_space)
        next_observation, reward, done, _ = env.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float64)
        if reward.shape != agent_return.shape:
            raise ValueError("environment reward must contain one value per agent")

        action_history.append(action.copy())
        observation_history.append(next_observation.copy())
        mode_codes.append(list(policy_info["mode_codes"]))
        mode_confidences.append(list(policy_info["mode_confidence"]))
        agent_return += reward
        update_return_to_go = getattr(policy, "update_return_to_go", None)
        if callable(update_return_to_go):
            update_return_to_go(reward)
        if capture_video:
            frames.append(_render_frame(env))
        if np.asarray(done).all():
            break

    return {
        "agent_return": agent_return,
        "team_return": float(agent_return.mean()),
        "steps": len(action_history),
        "observations": np.stack(observation_history),
        "actions": np.stack(action_history),
        "mode_codes": mode_codes,
        "mode_confidences": mode_confidences,
        "frames": np.stack(frames) if capture_video else None,
    }


class ModeOnlineEvaluator(_ModeEvaluatorBase):
    """加载 stage2 checkpoint，在 ``simple_spread`` 中真实 rollout。"""

    def init(
        self,
        log_dir: str,
        num_eval: int = 10,
        environment: str = "simple_spread-expert",
        max_episode_steps: int = 25,
        num_videos: int = 3,
        video_fps: int = 6,
        eval_seed: int = 1000,
        test_ret: Optional[float] = 0.9,
        use_return_to_go: bool = False,
        **kwargs,
    ):
        if "-" not in environment or environment.split("-", 1)[0] != "simple_spread":
            raise ValueError(
                "ModeOnlineEvaluator currently supports simple_spread only"
            )
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if not 0 <= num_videos <= num_eval:
            raise ValueError("num_videos must be between zero and num_eval")
        if video_fps < 1:
            raise ValueError("video_fps must be positive")

        super().init(
            log_dir=log_dir,
            num_eval=num_eval,
            dataset_split="train",
            **kwargs,
        )
        dataset = self.dataset
        self.normalizer = dataset.normalizer
        self.returns_scale = dataset.returns_scale
        self.discount = dataset.discount
        self.dataset_summary = {
            "dataset_path": dataset.dataset_path,
            "episodes": int(len(dataset.episode_indices)),
            "joint_environment_steps": int(
                dataset.store.mask[dataset.episode_indices].sum()
            ),
            "normalizer": dataset.normalizer.normalizer_name,
            "returns_scale": self.returns_scale,
            "discount": self.discount,
        }
        self.dataset = None
        del dataset
        gc.collect()

        self.environment = environment
        self.max_episode_steps = max_episode_steps
        self.num_videos = num_videos
        self.video_fps = video_fps
        self.eval_seed = eval_seed
        self.test_ret = test_ret
        self.use_return_to_go = bool(use_return_to_go)

    @torch.no_grad()
    def evaluate(self, load_step: int, **kwargs):
        del kwargs
        if not self.initialized:
            raise RuntimeError("evaluator.init must be called first")
        diffusion, checkpoint_path = self._load_objective(load_step)
        if self.condition_guidance_w is not None:
            diffusion.condition_guidance_w = self.condition_guidance_w
        if self.mode_guidance_w is not None:
            diffusion.mode_guidance_w = self.mode_guidance_w
        if self.use_ddim_sample:
            diffusion.set_ddim_scheduler(self.n_ddim_steps)

        policy = ModeOnlinePolicy(
            diffusion=diffusion,
            normalizer=self.normalizer,
            device=self.device,
            target_return=self.test_ret,
            returns_scale=self.returns_scale,
            discount=self.discount,
            use_return_to_go=self.use_return_to_go,
            local_confidence_threshold=self.local_confidence_threshold,
        )
        episode_results = []
        video_paths = []
        episode_seeds = []
        video_run = "step_{}".format(load_step)
        if self.use_ddim_sample:
            video_run += "-ddim"
        if self.condition_guidance_w is not None:
            video_run += "-cg_{}".format(self.condition_guidance_w)
        if self.mode_guidance_w is not None:
            video_run += "-mg_{}".format(self.mode_guidance_w)
        if self.use_return_to_go:
            video_run += "-rtg"
        for episode_index in range(self.num_eval):
            episode_seed = self.eval_seed + episode_index
            episode_seeds.append(episode_seed)
            set_seed(episode_seed)
            env = load_environment(self.environment, render_mode="rgb_array")
            try:
                episode = rollout_mpe_episode(
                    env=env,
                    policy=policy,
                    max_episode_steps=self.max_episode_steps,
                    capture_video=episode_index < self.num_videos,
                )
            finally:
                env.close()
            if episode["frames"] is not None:
                video_path = (
                    self.log_dir
                    / "videos"
                    / video_run
                    / "episode_{:03d}.mp4".format(episode_index)
                )
                save_video(
                    str(video_path),
                    episode.pop("frames"),
                    fps=self.video_fps,
                )
                video_paths.append(str(video_path))
            episode_results.append(episode)

        agent_returns = np.stack(
            [episode["agent_return"] for episode in episode_results]
        )
        team_returns = np.asarray(
            [episode["team_return"] for episode in episode_results]
        )
        mode_steps = [
            np.asarray(codes, dtype=np.int64)
            for episode in episode_results
            for codes in episode["mode_codes"]
        ]
        confidence_steps = [
            confidence
            for episode in episode_results
            for confidence in episode["mode_confidences"]
        ]
        full_coverage = [bool(np.all(codes >= 0)) for codes in mode_steps]
        agreement = [
            covered and bool(np.all(codes == codes[0]))
            for codes, covered in zip(mode_steps, full_coverage)
        ]
        result = {
            "status": "online_evaluation_complete",
            "review_status": "pending_user_discussion",
            "checkpoint": str(checkpoint_path),
            "environment_rollout": True,
            "environment": self.environment,
            "episode_seeds": episode_seeds,
            "num_episodes": self.num_eval,
            "max_episode_steps": self.max_episode_steps,
            "target_return_scaled": self.test_ret,
            "target_return_unscaled": (
                None
                if self.test_ret is None
                else float(self.test_ret * self.returns_scale)
            ),
            "returns_scale": self.returns_scale,
            "discount": self.discount,
            "use_return_to_go": self.use_return_to_go,
            "return_condition_definition": (
                "discounted_team_return_to_episode_end_divided_by_returns_scale"
            ),
            "return_definition": "mean_per_agent_environment_return",
            "average_team_return": float(team_returns.mean()),
            "std_team_return": float(team_returns.std()),
            "average_agent_return": agent_returns.mean(axis=0).tolist(),
            "std_agent_return": agent_returns.std(axis=0).tolist(),
            "episode_team_returns": team_returns.tolist(),
            "episode_agent_returns": agent_returns.tolist(),
            "local_mode_known_rate": (
                float(np.mean(np.concatenate(mode_steps) >= 0)) if mode_steps else 0.0
            ),
            "local_mode_full_coverage_rate": (
                float(np.mean(full_coverage)) if full_coverage else 0.0
            ),
            "local_mode_agreement_rate": (
                float(np.mean(agreement)) if agreement else 0.0
            ),
            "local_mode_mean_confidence": (
                float(np.mean(confidence_steps)) if confidence_steps else 0.0
            ),
            "episode_mode_codes": [
                episode["mode_codes"] for episode in episode_results
            ],
            "video_paths": video_paths,
            "training_dataset": self.dataset_summary,
        }
        self._save(load_step, result)
        return result


def _empty_action_history(env, n_agents: int) -> np.ndarray:
    action_dims = [int(np.prod(space.shape)) for space in env.action_space]
    if len(set(action_dims)) != 1 or len(action_dims) != n_agents:
        raise ValueError("all MPE agents must use the same vector action dimension")
    return np.empty((0, n_agents, action_dims[0]), dtype=np.float32)


def _clip_actions(actions: np.ndarray, action_spaces: Sequence) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    if len(action_spaces) != len(actions):
        raise ValueError("action count does not match environment agents")
    for agent, space in enumerate(action_spaces):
        if actions[agent].shape != space.shape:
            raise ValueError("policy action shape does not match MPE action space")
        actions[agent] = np.clip(actions[agent], space.low, space.high)
    return actions


def _render_frame(env) -> np.ndarray:
    frame = np.asarray(env.render())
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError("MPE render must return an RGB or RGBA frame")
    return frame[..., :3].astype(np.uint8)
