from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization

def make_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class CnnActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        critic_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        state_dim: int = 12,
        depth_height: int = 64,
        depth_width: int = 64,
        depth_feature_dim: int = 128,
        state_feature_dim: int = 64,
        **kwargs: dict[str, Any], 
    ) -> None:
        super().__init__()

        if kwargs:
            print("CnnActorCritic got unused kawrgs:", list(kwargs.keys()))
        if state_dependent_std:
            raise NotImplementedError("CnnActorCritic first version only supports state_dependent_std=False.")
        if noise_std_type != "scalar":
            raise NotImplementedError("CnnActorCritic first version only supports noise_std_type='sclar'.")


        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.depth_dim = depth_height * depth_width

        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        expected_obs_dim = self.state_dim + self.depth_dim

        if num_actor_obs != expected_obs_dim:
            raise ValueError(f"Actor obs dim is {num_actor_obs}, expected {expected_obs_dim}.")
        if num_critic_obs != expected_obs_dim:
            raise ValueError(f"Critic obs dim is {num_critic_obs}, expected {expected_obs_dim}.")

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        act = make_activation(activation)

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_depth = torch.zeros(1, 1, depth_height, depth_width)
            cnn_flat_dim = self.depth_encoder(dummy_depth).shape[-1]

        self.depth_projector = nn.Sequential(
            nn.Linear(cnn_flat_dim, depth_feature_dim),
            act,
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_feature_dim),
            act,
        )

        fused_dim = depth_feature_dim + state_feature_dim

        self.actor = self._make_head(fused_dim, num_actions, actor_hidden_dims, activation)
        self.critic = self._make_head(fused_dim, 1, critic_hidden_dims, activation)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None

        Normal.set_default_validate_args(False)

        print(f"CnnActorCritic depth encoder: {self.depth_encoder}")
        print(f"CnnActorCritic actor: {self.actor}")
        print(f"CnnActorCritic critic: {self.critic}")

    def _make_head(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(make_activation(activation))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)
    
    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError
    
    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)
    
    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)
    
    def _encode(self, flat_obs: torch.Tensor) -> torch.Tensor:
        state = flat_obs[:, : self.state_dim]
        depth = flat_obs[:, self.state_dim :]
        depth = depth.reshape(-1, 1, self.depth_height, self.depth_width)

        depth_feature = self.depth_projector(self.depth_encoder(depth))
        state_feature = self.state_encoder(state)
        return torch.cat([depth_feature, state_feature], dim=-1)
    
    def _update_distribution(self, obs: TensorDict) -> None:
        flat_obs = self.get_actor_obs(obs)
        flat_obs = self.actor_obs_normalizer(flat_obs)
        fused_feature = self._encode(flat_obs)
        mean = self.actor(fused_feature)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()
    
    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        flat_obs = self.get_actor_obs(obs)
        flat_obs = self.actor_obs_normalizer(flat_obs)
        fused_feature = self._encode(flat_obs)
        return self.actor(fused_feature)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        flat_obs = self.get_critic_obs(obs)
        flat_obs = self.critic_obs_normalizer(flat_obs)
        fused_feature = self._encode(flat_obs)
        return self.critic(fused_feature)
    
    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))
