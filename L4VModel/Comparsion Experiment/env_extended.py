"""
env_extended.py

Extended differentiable AAV-comm environment used for the TMC resubmission.
Adds, on top of the original static/equal-split/deterministic model in
dso_optimization.py:

  1. A learned per-user OFDMA bandwidth/power allocation a[t] (softmax head,
     sum_i a_i[t] = 1), replacing the fixed 1/N split. This introduces genuine
     resource competition between users that the policy must learn to manage.
  2. A reparameterized stochastic channel: log-normal shadowing and, optionally,
     Rician small-scale fading. The noise terms are exogenous standard-normal
     samples multiplied/added deterministically into the channel gain, so the
     rate is still a differentiable, pathwise function of the AAV position and
     the policy parameters (the same reparameterization trick used for VAEs),
     which is what lets BPTT keep working under stochastic channels.
  3. The rotary-wing UAV propulsion power model of Zeng, Xu & Zhang (IEEE TWC 2019,
     "Energy Minimization for Wireless Communication With Rotary-Wing UAV"), used
     to track/optionally penalize flight energy alongside completion time.

The original dso_optimization.py / L4V.py files are left untouched so that the
paper's originally reported figures remain reproducible from this repository.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Environment initialization (identical protocol to dso_optimization.py)
# ---------------------------------------------------------------------------
def initialize_environment(num_users, area_size):
    """Randomize user positions in [-area_size/2, area_size/2]^2 and task sizes in [2,4)."""
    user_positions = (torch.rand((num_users, 2), device=device) - 0.5) * area_size
    user_tasks = torch.rand(num_users, device=device) * 2 + 2
    base_station_pos = torch.zeros((1, 2), device=device)
    return user_positions, user_tasks, base_station_pos


# ---------------------------------------------------------------------------
# Stochastic, reparameterized channel model
# ---------------------------------------------------------------------------
def compute_transmission_rate_stochastic(base_pos, user_positions, h_unit, sigma,
                                          shadow_std=0.0, rician_k=None, generator=None):
    """Per-user achievable rate under LoS path loss plus reparameterized fading.

    shadow_std: standard deviation (in nepers) of log-normal shadowing. 0 disables it
        and recovers the original deterministic channel exactly.
    rician_k: Rician K-factor for small-scale fading. None disables it.

    Both noise terms are drawn as standard-normal samples and combined with the
    deterministic path-gain term through elementary differentiable operations
    (exp, square, multiply), so d(rate)/d(base_pos) is unaffected by the presence
    of the noise: the reparameterization trick keeps the adjoint gradient exact
    with respect to the policy parameters even though the channel realization
    is stochastic.
    """
    distances_squared = torch.sum((base_pos - user_positions) ** 2, dim=1)
    distances_squared = torch.clamp(distances_squared, min=0.1)
    path_gain = h_unit / distances_squared

    num_users = user_positions.shape[0]

    if shadow_std is not None and shadow_std > 0:
        eps_shadow = torch.randn(num_users, device=base_pos.device, generator=generator)
        shadow = torch.exp(shadow_std * eps_shadow)
    else:
        shadow = torch.ones(num_users, device=base_pos.device)

    if rician_k is not None:
        eps_r = torch.randn(num_users, device=base_pos.device, generator=generator)
        eps_i = torch.randn(num_users, device=base_pos.device, generator=generator)
        los_amp = math.sqrt(rician_k / (rician_k + 1.0))
        nlos_std = math.sqrt(1.0 / (2.0 * (rician_k + 1.0)))
        fading = (los_amp + nlos_std * eps_r) ** 2 + (nlos_std * eps_i) ** 2
    else:
        fading = torch.ones(num_users, device=base_pos.device)

    h = path_gain * shadow * fading
    rates = torch.log(1 + h / sigma)
    return rates


# ---------------------------------------------------------------------------
# Kinematics (unchanged)
# ---------------------------------------------------------------------------
def g(base_pos, angle, speed):
    dx = (torch.cos(angle) * speed).view(-1, 1)
    dy = (torch.sin(angle) * speed).view(-1, 1)
    movement = torch.cat([dx, dy], dim=-1).view_as(base_pos)
    return base_pos + movement


# ---------------------------------------------------------------------------
# UAV rotary-wing propulsion power model (Zeng, Xu & Zhang, TWC 2019)
# ---------------------------------------------------------------------------
def propulsion_power(speed, v_scale=1.0, P0=79.86, Pi=88.63, U_tip=120.0, v0=4.03,
                      d0=0.6, rho=1.225, s_solidity=0.05, A_rotor=0.503):
    """Closed-form power consumption P(v) for a rotary-wing UAV.

    Constants parameterize the model of Zeng, Xu & Zhang (2019); v_scale
    maps the simulator's normalized speed units to m/s for a physically meaningful
    energy number. speed may be a Python float or a torch tensor.
    """
    if not torch.is_tensor(speed):
        speed = torch.tensor(float(speed))
    v_phys = speed * v_scale
    blade_profile = P0 * (1.0 + 3.0 * v_phys ** 2 / U_tip ** 2)
    induced = Pi * torch.sqrt(
        torch.sqrt(1.0 + v_phys ** 4 / (4.0 * v0 ** 4)) - v_phys ** 2 / (2.0 * v0 ** 2)
    )
    parasite = 0.5 * d0 * rho * s_solidity * A_rotor * v_phys ** 3
    return blade_profile + induced + parasite


# ---------------------------------------------------------------------------
# Step loss with learned OFDMA allocation
# ---------------------------------------------------------------------------
def f_extended(base_pos, user_positions, remaining_tasks, alloc, h_unit, sigma,
               time_step, area_size, shadow_std=0.0, rician_k=None, generator=None):
    """One simulation step: stochastic rate, allocation-weighted transmission, backlog update.

    alloc is a length-N simplex vector (sums to 1). Scaling by num_users means that
    a uniform allocation (alloc_i = 1/N for all i) reduces exactly to the original
    equal-split model, so this environment strictly generalizes the paper's baseline
    rather than replacing it.
    """
    num_users = user_positions.shape[0]
    rates = compute_transmission_rate_stochastic(
        base_pos, user_positions, h_unit, sigma,
        shadow_std=shadow_std, rician_k=rician_k, generator=generator,
    )
    transmitted = alloc * num_users * rates * time_step
    new_remaining_tasks = torch.clamp(remaining_tasks - transmitted, min=0)

    task_loss = torch.sum(new_remaining_tasks)
    position_penalty = 0.01 * torch.sum(torch.abs(base_pos) - area_size / 2).clamp(min=0)
    loss = task_loss + position_penalty
    return loss, new_remaining_tasks, rates


# ---------------------------------------------------------------------------
# Policy network with a third, allocation, output head
# ---------------------------------------------------------------------------
class PathPlanningNetExtended(nn.Module):
    """Same trunk/heads as PathPlanningNet in dso_optimization.py, plus a softmax
    allocation head. The softmax enforces sum_i a_i = 1, a_i >= 0 by construction,
    exactly as the sigmoid head enforces the speed box constraint."""

    def __init__(self, num_users, v_max=0.2):
        super().__init__()
        input_dim = 2 + num_users * 2 + num_users
        self.num_users = num_users

        self.fc_1 = nn.Linear(input_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, 32)

        self.angle_head = nn.Linear(32, 1)
        self.speed_head = nn.Linear(32, 1)
        self.alloc_head = nn.Linear(32, num_users)

        self.v_max = v_max

        nn.init.xavier_uniform_(self.fc_1.weight)
        nn.init.xavier_uniform_(self.fc_2.weight)
        nn.init.xavier_uniform_(self.fc_3.weight)
        nn.init.uniform_(self.angle_head.weight, -0.01, 0.01)
        nn.init.uniform_(self.speed_head.weight, -0.01, 0.01)
        nn.init.uniform_(self.alloc_head.weight, -0.01, 0.01)

    def forward(self, base_pos, user_positions, remaining_tasks, use_allocation=True):
        relative_positions = user_positions - base_pos
        x = torch.cat([base_pos.flatten(), relative_positions.flatten(), remaining_tasks], dim=0)

        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        z = F.relu(self.fc_3(x))

        angle_preact = self.angle_head(z)
        angle = torch.tanh(angle_preact) * torch.pi

        speed_preact = self.speed_head(z)
        speed = torch.sigmoid(speed_preact) * self.v_max

        if use_allocation:
            alloc_logits = self.alloc_head(z)
            alloc = F.softmax(alloc_logits, dim=-1)
        else:
            alloc = torch.full((self.num_users,), 1.0 / self.num_users, device=z.device)

        return angle, speed, alloc, angle_preact
