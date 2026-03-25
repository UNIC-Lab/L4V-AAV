import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import matplotlib
matplotlib.use('TkAgg')
# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入各个优化方法
from dso_optimization import dso_optimize
from ga_optimization import ga_optimize
from dqn_optimization import dqn_optimize
from a2c_optimization import a2c_optimize
from ddpg_optimization import ddpg_optimize

# 定义方法名称
methods = ['DSO', 'GA', 'DQN', 'A2C', 'DDPG']
method_functions = [dso_optimize, ga_optimize, dqn_optimize, a2c_optimize, ddpg_optimize]

# 定义实验组
experiments = [
    {
        'name': 'Number of Users',
        'param_name': 'num_users',
        'param_values': [2, 4, 6, 8, 10],
        'fixed_params': {
            'area_size': 10.0,
            'h_unit': 1.0,
            'sigma': 0.1,
            'max_step': 0.2
        }
    },
    {
        'name': 'Area Size',
        'param_name': 'area_size',
        'param_values': [5.0, 10.0, 15.0, 20.0, 25.0],
        'fixed_params': {
            'num_users': 4,
            'h_unit': 1.0,
            'sigma': 0.1,
            'max_step': 0.2
        }
    },
    {
        'name': 'Unit Distance Gain',
        'param_name': 'h_unit',
        'param_values': [0.5, 1.0, 1.5, 2.0, 2.5],
        'fixed_params': {
            'num_users': 4,
            'area_size': 10.0,
            'sigma': 0.1,
            'max_step': 0.2
        }
    },
    {
        'name': 'Noise Power',
        'param_name': 'sigma',
        'param_values': [0.05, 0.10, 0.15, 0.20, 0.25],
        'fixed_params': {
            'num_users': 4,
            'area_size': 10.0,
            'h_unit': 1.0,
            'max_step': 0.2
        }
    }
]

# 定义指标名称
metrics = [
    'time_steps',
    'convergence_time',
    'convergence_episode',
    'avg_task_reduction'
]

metric_titles = {
    'time_steps': 'Average Time Steps',
    'convergence_time': 'Average Convergence Time (s)',
    'convergence_episode': 'Average Convergence Episode',
    'avg_task_reduction': 'Average Task Reduction per Step'
}


# 运行实验
def run_experiments():
    results = {}
    transmission_rates = {}  # 存储传输速率用于最后的直方图

    for exp_idx, experiment in enumerate(experiments):
        exp_name = experiment['name']
        param_name = experiment['param_name']
        param_values = experiment['param_values']
        fixed_params = experiment['fixed_params']

        print(f"\n=== Running Experiment {exp_idx + 1}: {exp_name} ===")

        # 初始化结果存储
        results[exp_name] = {}
        for metric in metrics:
            results[exp_name][metric] = {}
            for method in methods:
                results[exp_name][metric][method] = []

        # 初始化传输速率存储
        transmission_rates[exp_name] = {}
        for method in methods:
            transmission_rates[exp_name][method] = []

        # 遍历参数值
        for param_value in param_values:
            print(f"\n--- {param_name}: {param_value} ---")

            # 遍历方法
            for method_idx, method in enumerate(methods):
                print(f"Running {method}...")

                # 准备参数
                params = fixed_params.copy()
                params[param_name] = param_value
                params['time_step_duration'] = 0.1  # 固定时间步长

                # 运行方法20次
                method_results = {metric: [] for metric in metrics}
                method_transmission_rates = []

                for run_idx in range(10):
                    try:
                        result = method_functions[method_idx](**params)

                        # 记录指标
                        for metric in metrics:
                            method_results[metric].append(result[metric])

                        # 记录传输速率（只记录最后一次运行的）
                        if run_idx == 9:
                            method_transmission_rates.extend(result['transmission_rates'])

                    except Exception as e:
                        print(f"Error running {method} with {param_name}={param_value}: {e}")
                        # 添加默认值以避免中断
                        for metric in metrics:
                            method_results[metric].append(0)

                # 计算平均值并存储
                for metric in metrics:
                    results[exp_name][metric][method].append(np.mean(method_results[metric]))

                # 存储传输速率
                transmission_rates[exp_name][method].extend(method_transmission_rates)

                # 打印当前方法的平均结果
                print(f"{method} results:")
                for metric in metrics:
                    print(f"  {metric}: {np.mean(method_results[metric]):.4f}")

    return results, transmission_rates




# 打印实验结果
def print_results(results):
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    for exp_name in results:
        print(f"\n{exp_name}:")
        print("-" * 40)

        for metric in metrics:
            print(f"\n{metric_titles[metric]}:")
            for method in methods:
                avg_value = np.mean(results[exp_name][metric][method])
                print(f"  {method}: {avg_value:.4f}")


# 主函数
def main():
    print("Starting UAV Trajectory Optimization Experiments")
    print("This may take a while...")

    # 运行实验
    start_time = time.time()
    results, transmission_rates = run_experiments()
    total_time = time.time() - start_time

    print(f"\nTotal experiment time: {total_time:.2f} seconds")

    # 打印结果
    print_results(results)


if __name__ == "__main__":
    main()