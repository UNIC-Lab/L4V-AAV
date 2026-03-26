# Learn for Variation (L4V)

**Learn for Variation (L4V)** is a gradient‑informed trajectory learning framework for autonomous aerial vehicle ($AAV$) data collection in $6G \:\:IoT\:\:networks $.  It replaces high‑variance scalar reward signals with dense, analytically grounded policy gradients derived from a fully differentiable simulation environment. 

This repository contains the complete implementation of $L4V$ and the experimental code used in the paper:

> Xiucheng Wang, Zhenye Chen, Nan Cheng. *Learn for Variation: Variationally Guided AAV Trajectory Learning in Differentiable Environments*.
>   
> [arXiv link of the paper](https://arxiv.org/abs/2603.18853)
If there are any of your papers that have not been collected, please contact us
## ✨Key Features

**End‑to‑end differentiable simulation** The $AAV$ kinematics, channel‑dependent transmission rates, and data backlog evolution are unrolled into a differentiable computation graph.

**Exact variational gradients** Backpropagation through time ($BPTT$) implements a discrete adjoint solver, providing low‑variance, physically‑meaningful gradients.

**Deterministic neural policy** A lightweight $MLP$ outputs continuous speed and heading commands, trained via gradient descent on the global mission cost. 

**Regularization & stability** Temporal smoothness penalty and gradient clipping ensure physically plausible trajectories and stable long‑horizon training.

**Comprehensive baselines** Includes $GA, DQN, A2C, and DDPG$ implementations for fair comparison.

## 📁  Repository Structure
The directory structure of the engineering files is as follows:
```
L4VModel/
├── L4V.py Main L4V model and training loop
├── Comparison Experiment/
│ ├── dso_optimization.py # L4V Model
│ ├── ga_optimization.py # Genetic Algorithm baseline
│ ├── dqn_optimization.py # DQN baseline
│ ├── a2c_optimization.py # A2C baseline
│ ├── ddpg_optimization.py # DDPG baseline
│ ├── run_dso.py # Run L4V experiments
│ ├── run_ga.py # Run GA experiments
│ ├── run_dqn.py # Run DQN experiments
│ ├── run_a2c.py # Run A2C experiments
│ ├── run_ddpg.py # Run DDPG experiments
│ ├── top.py # Top‑level script to run all experiments
│ ├── total.py # Generate figures from saved results
│ └── results/
│   ├── data/ # CSV files with raw results
│   └── image/ # Generated figures
└── UAV_Sequence_Optimization.pdf # The full paper
```
## 🚀Quick Start

### 1. Clone the repository
```
git clone https://github.com/UNIC-Lab/L4V-AAV.git
cd L4VModel
```
### 2. Install dependencies
We recommend using ```Python 3.8+``` and a virtual environment.
```
pip install torch numpy matplotlib pandas deap
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
cd "Comparison Experiment"
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

##📊 Results Overview
The proposed L4V consistently outperforms all baselines:

Shorter mission completion time – lower median and variance across all scenarios.

Higher average transmission rate – better exploitation of channel geometry.

Faster training – deterministic gradients reduce the number of required iterations and wall‑clock time.

Detailed results can be found in the paper and reproduced using the code above.

## 📖Citation
If you use this code or find our work helpful, please cite:
```
bibtex
@article{wang2024learn,
  title={Learn for Variation: Variationally Guided AAV Trajectory Learning in Differentiable Environments},
  author={Wang, Xiucheng and Chen, Zhenye and Cheng, Nan},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

## 🙏Acknowledgements
We thank the UNIC research group for their strong support as well as the efforts of our colleagues and classmates.