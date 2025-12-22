import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim

try:
    # PyTorch 里用于 low-rank + diag 的多元高斯（推荐）
    from torch.distributions import LowRankMultivariateNormal
    _HAS_LR_MVN = True
except Exception:
    from torch.distributions import MultivariateNormal
    _HAS_LR_MVN = False


class _LatticeDist:
    """
    最小分布包装：满足 SB3 policy 的调用习惯：
    - get_actions(deterministic)
    - log_prob(actions)
    - entropy()
    """
    def __init__(self, dist):
        self.dist = dist

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            # mode/mean
            return self.dist.mean
        # 参照原实现：用 rsample() 便于梯度传播 :contentReference[oaicite:3]{index=3}
        return self.dist.rsample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()


class LatticeActorCriticPolicy(ActorCriticPolicy):
    """
    SB3 适配的 Lattice 高斯策略：
    Sigma = W diag(latent_var) W^T + diag(action_var)
    - W: action_net.weight (action_dim x latent_dim_pi)
    - log_std 参数长度: action_dim + latent_dim_pi（参照原实现）:contentReference[oaicite:4]{index=4}
    """

    def __init__(
        self,
        *args,
        lattice_init_log_std: float = -0.5,
        lattice_fix_std: bool = False,
        lattice_eps: float = 1e-6,
        **kwargs,
    ):
        self.lattice_init_log_std = float(lattice_init_log_std)
        self.lattice_fix_std = bool(lattice_fix_std)
        self.lattice_eps = float(lattice_eps)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        # 先让 SB3 正常构建：features_extractor / mlp_extractor / value_net / action_net 等
        super()._build(lr_schedule)

        # action_net 是线性层：latent_dim_pi -> action_dim
        # latent_dim_pi 由 SB3 的 mlp_extractor 决定
        action_dim = get_action_dim(self.action_space)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # 参照原 lattice：log_std 长度是 action_dim + latent_dim :contentReference[oaicite:5]{index=5}
        self.lattice_log_std = nn.Parameter(
            torch.ones(1, action_dim + latent_dim_pi) * self.lattice_init_log_std,
            requires_grad=(not self.lattice_fix_std),
        )

        # 参照原 lattice：action_mean 初始化更“温和”一些 :contentReference[oaicite:6]{index=6}
        if hasattr(self, "action_net") and isinstance(self.action_net, nn.Linear):
            with torch.no_grad():
                self.action_net.weight.mul_(0.1)
                if self.action_net.bias is not None:
                    self.action_net.bias.mul_(0.0)

    def _get_lattice_dist(self, mean_actions: torch.Tensor) -> _LatticeDist:
        """
        依据当前 action_net.weight 和 lattice_log_std 构造分布
        """
        action_dim = mean_actions.shape[-1]
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        std = torch.exp(self.lattice_log_std)  # (1, action_dim + latent_dim_pi)
        action_var = std[:, :action_dim].pow(2).squeeze(0)          # (action_dim,)
        latent_var = std[:, action_dim:].pow(2).squeeze(0)          # (latent_dim_pi,)

        # 参照原实现：sigma_mat = (W * latent_var).matmul(W^T) + diag(action_var) :contentReference[oaicite:7]{index=7}
        # 这里用 LowRankMultivariateNormal 更高效：cov = F F^T + diag(cov_diag)
        W = self.action_net.weight  # (action_dim, latent_dim_pi)
        # cov_factor: (action_dim, rank), rank=latent_dim_pi
        cov_factor = W * torch.sqrt(latent_var).unsqueeze(0)  # (action_dim, latent_dim_pi)
        cov_diag = action_var + self.lattice_eps

        if _HAS_LR_MVN:
            # 广播到 batch：mean_actions 为 (batch, action_dim)
            batch = mean_actions.shape[0]
            cov_factor_b = cov_factor.unsqueeze(0).expand(batch, -1, -1)  # (batch, action_dim, rank)
            cov_diag_b = cov_diag.unsqueeze(0).expand(batch, -1)          # (batch, action_dim)
            dist = LowRankMultivariateNormal(loc=mean_actions, cov_factor=cov_factor_b, cov_diag=cov_diag_b)
        else:
            # 兜底：显式构造 full covariance（action_dim 大时会慢/占显存）
            sigma = cov_factor @ cov_factor.t() + torch.diag(cov_diag)
            dist = MultivariateNormal(loc=mean_actions, covariance_matrix=sigma)

        return _LatticeDist(dist)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> _LatticeDist:
        mean_actions = self.action_net(latent_pi)
        return self._get_lattice_dist(mean_actions)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._get_action_dist_from_latent(latent_pi)

        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._get_action_dist_from_latent(latent_pi)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy
