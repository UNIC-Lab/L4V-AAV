"""DQN baseline under the same reparameterized stochastic channel as the
extended L4V/MPC environment (see stochastic_patch.py). Action space, network,
and training loop are otherwise identical to dqn_optimization.py.
"""
import dqn_optimization as _dqn
from stochastic_patch import patch_agent_class

patch_agent_class(_dqn.DQNAgent)

dqn_optimize_extended = _dqn.dqn_optimize
