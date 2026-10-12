"""DDPG baseline under the same reparameterized stochastic channel as the
extended L4V/MPC environment (see stochastic_patch.py). Action space, network,
and training loop are otherwise identical to ddpg_optimization.py.
"""
import ddpg_optimization as _ddpg
from stochastic_patch import patch_agent_class

patch_agent_class(_ddpg.DDPGAgent)

ddpg_optimize_extended = _ddpg.ddpg_optimize
