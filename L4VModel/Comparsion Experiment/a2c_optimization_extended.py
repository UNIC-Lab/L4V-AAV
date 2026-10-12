"""A2C baseline under the same reparameterized stochastic channel as the
extended L4V/MPC environment (see stochastic_patch.py). Action space, network,
and training loop are otherwise identical to a2c_optimization.py.
"""
import a2c_optimization as _a2c
from stochastic_patch import patch_agent_class

patch_agent_class(_a2c.A2CAgent)

a2c_optimize_extended = _a2c.a2c_optimize
