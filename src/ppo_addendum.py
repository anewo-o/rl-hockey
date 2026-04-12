from __future__ import annotations
import warnings
from typing import Any, ClassVar, TypeVar, Generator, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutBufferSamples
from stable_baselines3.common.utils import FloatSchedule, explained_variance

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PartialRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer that computes standard SB3 truncated GAE, 
    then masks out the high-bias tail of unfinished trajectory segments.

    Paper rule implemented:
    - if a sampled segment ends because the episode ended inside the buffer:
        use all transitions from that segment
    - if a sampled segment reaches the end of the rollout buffer without termination:
        use only the first `partial_horizon` transitions from that segment
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        partial_horizon: Optional[int] = None,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )
        self.partial_horizon = partial_horizon if partial_horizon is not None else buffer_size
        if not (1 <= self.partial_horizon <= buffer_size):
            raise ValueError(
                f"partial_horizon must be in [1, buffer_size], got {self.partial_horizon}"
            )
        self.valid_mask: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.valid_mask = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        # Standard SB3 truncated GAE.
        super().compute_returns_and_advantage(last_values, dones)

        # Build a mask over valid training samples.
        # A "segment" is a contiguous chunk within one env between episode starts.
        # If the segment ends inside the buffer, it is complete => keep all.
        # If it reaches the buffer end, it is incomplete => keep only prefix of length partial_horizon.
        self.valid_mask[:] = False

        for env_idx in range(self.n_envs):
            starts = [0]
            # episode_starts[t] == 1 means obs at t is the first obs of a new episode
            for t in range(1, self.buffer_size):
                if self.episode_starts[t, env_idx] > 0.5:
                    starts.append(t)

            for seg_i, seg_start in enumerate(starts):
                seg_end = starts[seg_i + 1] if seg_i + 1 < len(starts) else self.buffer_size
                segment_is_complete = seg_end < self.buffer_size

                if segment_is_complete:
                    self.valid_mask[seg_start:seg_end, env_idx] = True
                else:
                    keep_end = min(seg_start + self.partial_horizon, seg_end)
                    self.valid_mask[seg_start:keep_end, env_idx] = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""

        if not self.generator_ready:
            tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "valid_mask",
            ]
            for name in tensor_names:
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True

        valid_indices = np.flatnonzero(self.valid_mask)
        if len(valid_indices) == 0:
            raise RuntimeError(
                "PartialGAERolloutBuffer produced zero valid samples. "
                "Increase partial_horizon or n_steps."
            )

        indices = np.random.permutation(valid_indices)

        if batch_size is None:
            batch_size = len(indices)
        else:
            batch_size = min(batch_size, len(indices))

        start_idx = 0
        while start_idx < len(indices):
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size


class PartialPPO(PPO):
    """
    PPO + partial GAE masking.

    Reuses SB3 normal:
    - learn()
    - collect_rollouts()
    - train()

    Customization is in the rollout buffer class.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy,
        env: GymEnv | str,
        *,
        partial_horizon: Optional[int] = None,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: Optional[float | Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs=None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str | th.device = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if rollout_buffer_class is not None:
            raise ValueError(
                "PartialPPO manages rollout_buffer_class itself. "
                "Pass extra args through rollout_buffer_kwargs if needed."
            )

        rollout_buffer_kwargs = dict(rollout_buffer_kwargs or {})
        rollout_buffer_kwargs["partial_horizon"] = (
            n_steps if partial_horizon is None else partial_horizon
        )

        self.partial_horizon = rollout_buffer_kwargs["partial_horizon"]

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=PartialRolloutBuffer,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self) -> None:
        """
        From original SB3, only here for tracking.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
