"""
从 results/data 文件夹中加载：
  <method>_result_<param>_<value>.csv

绘制：
  - 提琴图：任务完成时间（time_steps）与平均传输速率（avg_task_reduction）分布
  - 柱状图：训练轮次（convergence_episode）与训练时间（total_training_time）均值 ± 四分位点
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===============================
# 基本设置
# ===============================
RESULTS_DIR = "results/data"
METHODS = ["dso", "ga", "dqn", "a2c", "ddpg"]
COLORS = {
    "dso": "#1f77b4",
    "ga": "#ff7f0e",
    "dqn": "#2ca02c",
    "a2c": "#d62728",
    "ddpg": "#9467bd"
}

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


# ===============================
# 读取所有结果文件
# ===============================
def load_results(result_dir=RESULTS_DIR):
    """
    从 results/ 目录加载所有匹配格式的 csv
    返回 results[method][param][value] = DataFrame
    """
    results = {}
    pattern = re.compile(r"(?P<method>dso|ga|dqn|a2c|ddpg)_result_(?P<param>\w+)_(?P<val>[-+]?\d*\.?\d+)\.csv")

    for file in os.listdir(result_dir):
        m = pattern.match(file)
        if m:
            method = m.group("method")
            param = m.group("param")
            val = float(m.group("val"))
            df = pd.read_csv(os.path.join(result_dir, file))
            results.setdefault(method, {}).setdefault(param, {})[val] = df
    return results


# ===============================
# 绘图函数：提琴图（分布）
# ===============================
def plot_violin(metric, ylabel, results, param_name, save_path):
    """
    绘制提琴图分布（time_steps, avg_task_reduction）
    每个参数值在 x 轴，5 种方法分组显示
    """
    print(f"🎻 绘制 {metric} 提琴图...")

    fig, ax = plt.subplots(figsize=(8, 8))  # 正方形图形

    # 获取参数值
    param_values = sorted(list(next(iter(results.values()))[param_name].keys()))
    n_params = len(param_values)
    n_methods = len(METHODS)

    group_width = 0.8
    method_width = group_width / n_methods * 0.9
    centers = np.arange(n_params)
    offsets = np.linspace(-group_width/2 + method_width/2, group_width/2 - method_width/2, n_methods)

    legend_handles = []

    # 收集所有数据以确定y轴范围
    all_data = []

    for mi, method in enumerate(METHODS):
        if method not in results:
            continue
        positions = centers + offsets[mi]
        data_per_param = [
            results[method][param_name][v][metric].dropna().values
            if v in results[method][param_name] else []
            for v in param_values
        ]

        for i, data_i in enumerate(data_per_param):
            if len(data_i) == 0:
                continue
            vp = ax.violinplot([data_i], positions=[positions[i]], widths=method_width,
                               showmeans=False, showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor(COLORS[method])
                pc.set_edgecolor("black")
                pc.set_alpha(0.6)

            # 平均值点
            ax.scatter(positions[i], np.mean(data_i), color="k", s=25, marker="D")

            all_data.extend(data_i)
        legend_handles.append(Patch(facecolor=COLORS[method], edgecolor="black", label="L4V" if method == "dso" else method.upper()))

    ax.set_xticks(centers)
    ax.set_xticklabels([str(v) for v in param_values], fontname='Times New Roman')
    ax.set_xlabel(param_name, fontsize=14, fontname='Times New Roman')
    ax.set_ylabel(ylabel, fontsize=14, fontname='Times New Roman')
    ax.grid(axis="y", alpha=0.3)

    # ⭐⭐⭐ 这里是关键改动：增加 y 上界，让 legend 不会挡住图形
    if all_data:
        y_min = np.min(all_data)
        y_max = np.max(all_data)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.325 * y_range)

    ax.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=14,  # ← 字体更大
        prop={'family': 'Times New Roman'},
        frameon=True,
        markerscale=1.6,  # ← 图例中颜色块更大
        borderpad=0.8,  # ← 图例框内填充更大
        labelspacing=0.6  # ← 图例条目之间更大间距
    )

    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ===============================
# 绘图函数：柱状图（均值 ± 四分位点）
# ===============================
def plot_bar(metric, ylabel, results, param_name, save_path):
    """
    绘制柱状图（训练时间、训练轮次）
    误差线表示上下四分位点（25%和75%分位数）
    """
    print(f"📊 绘制 {metric} 柱状图...")

    fig, ax = plt.subplots(figsize=(8, 8))  # 正方形图形

    param_values = sorted(list(next(iter(results.values()))[param_name].keys()))
    n_params = len(param_values)
    n_methods = len(METHODS)

    group_width = 0.8
    bar_width = group_width / n_methods * 0.9
    centers = np.arange(n_params)
    offsets = np.linspace(-group_width/2 + bar_width/2, group_width/2 - bar_width/2, n_methods)

    for mi, method in enumerate(METHODS):
        if method not in results:
            continue
        means, lower_errors, upper_errors = [], [], []
        for v in param_values:
            if v in results[method][param_name]:
                vals = results[method][param_name][v][metric].dropna().values
                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    q1 = np.percentile(vals, 25)
                    q3 = np.percentile(vals, 75)

                    means.append(mean_val)
                    lower_error = abs(mean_val - q1)
                    upper_error = abs(q3 - mean_val)
                    lower_errors.append(lower_error)
                    upper_errors.append(upper_error)
                else:
                    means.append(np.nan)
                    lower_errors.append(0)
                    upper_errors.append(0)
            else:
                means.append(np.nan)
                lower_errors.append(0)
                upper_errors.append(0)

        pos = centers + offsets[mi]

        valid_indices = ~np.isnan(means)
        if np.any(valid_indices):
            valid_pos = pos[valid_indices]
            valid_means = np.array(means)[valid_indices]
            valid_lower = np.array(lower_errors)[valid_indices]
            valid_upper = np.array(upper_errors)[valid_indices]

            ax.bar(valid_pos, valid_means, width=bar_width,
                   yerr=[valid_lower, valid_upper],
                   capsize=4, color=COLORS[method], alpha=0.8,
                   label="L4V" if method == "dso" else method.upper(), error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    ax.set_xticks(centers)
    ax.set_xticklabels([str(v) for v in param_values], fontname='Times New Roman')
    ax.set_xlabel(param_name, fontsize=14, fontname='Times New Roman')
    ax.set_ylabel(ylabel, fontsize=14, fontname='Times New Roman')
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        loc='upper right',
        fontsize=14,  # ← 字体更大
        prop={'family': 'Times New Roman'},
        frameon=True,
        markerscale=1.6,  # ← 图例中颜色块更大
        borderpad=0.8,
        labelspacing=0.6
    )

    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

    # ⭐⭐⭐ 新增：扩大 y 轴高度，避免 legend 挡住柱状图
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ===============================
# 主入口
# ===============================
if __name__ == "__main__":
    results = load_results(RESULTS_DIR)
    if not results:
        print("⚠️ 没有找到任何CSV文件，请先运行 run_xxx.py 生成数据！")
        exit(0)

    sample_method = next(iter(results))
    param_name = "num_users"

    print(f"检测到参数: {param_name}")
    print(f"检测到方法: {', '.join(results.keys())}")

    plot_violin("time_steps", "Time Steps", results, param_name,
                os.path.join(RESULTS_DIR, f"violin_time_steps_{param_name}.png"))

    plot_violin("avg_task_reduction", "Average Transmission Rate", results, param_name,
                os.path.join(RESULTS_DIR, f"violin_avg_rate_{param_name}.png"))

    plot_bar("convergence_episode", "Convergence Episode", results, param_name,
             os.path.join(RESULTS_DIR, f"bar_convergence_episode_{param_name}.png"))

    plot_bar("total_training_time", "Training Time (s)", results, param_name,
             os.path.join(RESULTS_DIR, f"bar_training_time_{param_name}.png"))