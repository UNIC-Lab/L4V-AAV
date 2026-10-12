"""Shared stochastic-channel patch applied to the DQN/A2C/DDPG baselines so they
face the identical reparameterized log-normal-shadowing + Rician-fading channel
used by the extended L4V/MPC environment (env_extended.py), for a fair, richer
baseline roster in the noise-power sweep. These baselines have no allocation
action, so the rate uses the full (equal-split-equivalent) channel gain, exactly
matching each baseline's original deterministic formula with only the channel
term replaced.
"""
import math
import torch

DEFAULT_SHADOW_STD = 0.15
DEFAULT_RICIAN_K = 6.0


def make_stochastic_rate_fn(shadow_std=DEFAULT_SHADOW_STD, rician_k=DEFAULT_RICIAN_K):
    def compute_transmission_rate(self, base_pos, user_positions):
        distances_squared = torch.sum((base_pos - user_positions) ** 2, dim=1)
        distances_squared = torch.clamp(distances_squared, min=0.1)
        path_gain = self.h_unit / distances_squared

        num_users = user_positions.shape[0]
        dev = base_pos.device
        eps_shadow = torch.randn(num_users, device=dev)
        shadow = torch.exp(shadow_std * eps_shadow)

        eps_r = torch.randn(num_users, device=dev)
        eps_i = torch.randn(num_users, device=dev)
        los_amp = math.sqrt(rician_k / (rician_k + 1.0))
        nlos_std = math.sqrt(1.0 / (2.0 * (rician_k + 1.0)))
        fading = (los_amp + nlos_std * eps_r) ** 2 + (nlos_std * eps_i) ** 2

        h = path_gain * shadow * fading
        rates = torch.log(1 + h / self.sigma)
        return rates

    return compute_transmission_rate


def patch_agent_class(agent_cls, shadow_std=DEFAULT_SHADOW_STD, rician_k=DEFAULT_RICIAN_K):
    """Monkeypatch an agent class's compute_transmission_rate in place."""
    agent_cls.compute_transmission_rate = make_stochastic_rate_fn(shadow_std, rician_k)
    return agent_cls
