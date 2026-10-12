# Learn for Variation (L4V)

**Learn for Variation (L4V)** is a gradient-informed trajectory-learning framework for autonomous aerial vehicle (AAV) data collection in 6G IoT networks. It treats the coupled AAV kinematics, wireless channel, shared resource allocation, and residual demand as a structured differentiable wireless world model and uses conditional pathwise gradients to train a reusable policy.

This repository contains the complete implementation of $L4V$ and the experimental code used in the paper:

> Xiucheng Wang, Zhenye Chen, Nan Cheng, Zhisheng Yin, Xuemin Shen. *Learn for Variation: Efficient AAV Trajectory Learning through a Differentiable Wireless World Model*.
>
> [arXiv link of the paper](https://arxiv.org/abs/2603.18853)

## ✨Key Features

**Structured wireless world model.** AAV kinematics, stochastic channel-dependent rates, shared allocation, and backlog evolution are unrolled into a differentiable computation graph.

**Scoped pathwise gradients.** Backpropagation through time implements the discrete adjoint and returns the exact derivative of a fixed realized rollout. Stochastic expected-objective optimization still has sampling variance.

**Deterministic neural policy.** A 64--64--32 MLP outputs bounded displacement and heading together with a simplex-constrained OFDMA allocation.

**Regularization and stability.** Heading smoothness and gradient clipping improve trajectory regularity and long-horizon numerical stability.

**Baselines and stress tests.** The repository includes GA, DQN, A2C, DDPG, differentiable MPC, stochastic allocation, energy, channel-mismatch, partial-observation, multi-AAV, obstacle, smoothness, and zero-shot experiments.

## 📁  Repository Structure
The directory structure of the engineering files is as follows:
```
L4VModel/
├── L4V.py Main L4V model and training loop
├── Comparsion Experiment/ # Historical directory spelling retained for compatibility
│ ├── dso_optimization.py # L4V Model
│ ├── env_extended.py # Stochastic channel, allocation, and energy environment
│ ├── qcqp_optimization.py # SCA-QCQP reference planner
│ ├── ga_optimization.py # Genetic Algorithm baseline
│ ├── dqn_optimization.py # DQN baseline
│ ├── a2c_optimization.py # A2C baseline
│ ├── ddpg_optimization.py # DDPG baseline
│ ├── run_dso.py # Run L4V experiments
│ ├── run_ga.py # Run GA experiments
│ ├── run_dqn.py # Run DQN experiments
│ ├── run_a2c.py # Run A2C experiments
│ ├── run_ddpg.py # Run DDPG experiments
│ ├── run_surrogate_gap_revision.py # Exact P1/P3 finite-action audit
│ ├── run_qcqp_revision.py # SCA-QCQP experiments
│ ├── run_csi_adaptation_revision.py # Instantaneous-CSI adaptation
│ ├── run_high_speed_revision.py # Physically calibrated high-speed study
│ ├── run_stress_tests.py # Robustness and structural extensions
│ ├── run_zero_shot_generalization.py # Frozen-policy transfer
│ ├── top.py # Top‑level script to run all experiments
│ ├── total.py # Generate figures from saved results
│ └── results/
│   ├── data/ # CSV files with raw results
│   └── image/ # Generated figures
```
## 🚀Quick Start

### 1. Clone the repository
```
git clone https://github.com/UNIC-Lab/L4V-AAV.git
cd L4V-AAV
```
### 2. Install dependencies
We recommend Python 3.11 and a virtual environment.
```
pip install -r requirements.txt
```
```torch``` – For neural networks and automatic differentiation
```numpy``` – Numerical operations
```matplotlib``` – Plotting and visualisation
```pandas``` – CSV handling
```deap``` – Genetic Algorithm baseline
### 3. Run a single algorithm
Each ```run_*.py``` script runs a specific algorithm with a configurable parameter sweep.
Example – run $L4V (DSO)$ experiments for varying noise power:
```
cd "L4VModel/Comparsion Experiment"
python run_dso.py
```
Results are saved in results/data/ as CSV files named like ```dso_result_sigma_0.10.csv```.
### 4. Run all experiments
```
python top.py
```
This will sequentially run all five algorithms across all experimental settings (```number of users```, ```area size```, ```channel gain```, ```noise power```).
**Caution**: This may take several hours depending on your hardware.
### 5. Reproduce the paper figures
After the experiments finish, generate the violin and bar plots:
```
python total.py
```
The figures will be saved in ```results/image/```.

### 6. Reproduce the revision stress tests

From `L4VModel/Comparsion Experiment`, run:

```bash
python run_stress_tests.py --part all --smoke
python run_stress_tests.py --part robustness
python run_stress_tests.py --part multi
python run_stress_tests.py --part obstacle
python run_stress_tests.py --part smoothness
python run_zero_shot_generalization.py --smoke
python run_zero_shot_generalization.py
python plot_stress_tests.py
```

Aggregate NPZ files are written to `results_extended/stress/` and
`results_extended/generalization/`. Raw per-trial NPZ files retain trajectories,
objective histories, timings, seeds, convergence curves, and metadata. The full
zero-shot run pretrains one policy on 1,600 randomized layouts and evaluates it
without updates on 20 disjoint held-out layouts with three channel rollouts each.

### 7. Reproduce the second-round studies

The following smoke commands verify the revision pipelines with small workloads:

```bash
python run_surrogate_gap_revision.py --smoke
python run_qcqp_revision.py --smoke
python run_csi_adaptation_revision.py --smoke
python run_high_speed_revision.py --smoke
```

Remove `--smoke` to run the configured full study. Generated results, checkpoints,
plots, logs, and caches are excluded from new commits by `.gitignore`. Smoke mode
checks pipeline execution and does not reproduce the reported full-study statistics.

## 🧪 Experiments
The following factors are varied to evaluate performance:

| Experiment | Varying parameter | Fixed parameters |
|------------|-------------------|------------------|
| $User \:\:density$ | `num_users` ∈ {2,4,6,8,10} | `area_size=10`, `h_unit=1`, `sigma=0.1` |
| $Area\:\: size$ | `area_size` ∈ {5,10,15,20,25} | `num_users=4`, `h_unit=1`, `sigma=0.1` |
| $Channel\:\: gain$ | `h_unit` ∈ {0.5,1.0,1.5,2.0,2.5} | `num_users=4`, `area_size=10`, `sigma=0.1` |
| $Noise\:\: power$ | `sigma` ∈ {0.05,0.10,0.15,0.20,0.25} | `num_users=4`, `area_size=10`, `h_unit=1` |

The reported metrics are:

```time_steps``` – number of steps to complete the mission

```avg_task_reduction``` – average amount of data transmitted per step

```convergence_episode``` – iteration number when training stabilises

```total_training_time``` – wall‑clock time until convergence

## 📊 Results Overview

The paper reports mission completion, effective collection rate, compute cost,
allocation and energy ablations, executed trajectories, and reviewer-driven
stress tests. Capped runs, regressions, model-mismatch degradation, and obstacle
collisions are retained rather than discarded. After pretraining on 1,600
randomized layouts, the frozen zero-shot policy completes all 60 executed
rollouts on 20 disjoint held-out layouts within the sampled layout/channel
distribution.

## 📖Citation
If you use this code or find our work helpful, please cite:
```
bibtex
@article{wang2026l4v,
  title={Learn for Variation: Efficient AAV Trajectory Learning through a Differentiable Wireless World Model},
  author={Xiucheng Wang, Zhenye Chen, Nan Cheng, Zhisheng Yin, Xuemin Shen},
  journal={arXiv preprint arXiv:2603.18853},
  year={2026}
}
```

## 🙏Acknowledgements
We thank the UNIC research group for their strong support as well as the efforts of our colleagues and classmates.
