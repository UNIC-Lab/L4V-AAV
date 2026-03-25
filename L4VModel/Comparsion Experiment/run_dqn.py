"""
run_dso.py
运行 DSO 优化算法实验，针对一个参数组（如 num_users）变化进行多次实验。
每组实验重复多次，结果保存为 CSV，并打印该组平均值+标准差。
"""

import os
import numpy as np
import pandas as pd
from dqn_optimization import dqn_optimize

# ==========================================
# 实验配置
# ==========================================

# 选择变化的参数（可以改为 'area_size', 'h_unit', 'sigma'）
param_name = "area_size"
param_values = [10,15,20,25]

# 固定参数
fixed_params = {
    "sigma": 0.1,
    "num_users": 4,
    "h_unit": 1.0,
    "max_step": 0.2,
    "time_step_duration": 0.1
}

# 每组重复实验次数
n_repeat = 10

# 输出目录
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 方法名称（用于命名）
method_name = "dqn"


# ==========================================
# 实验运行函数
# ==========================================

def run_dso_experiments():
    print(f"🚁 开始运行 DQN 实验")
    print(f"变化参数: {param_name}")
    print(f"参数取值: {param_values}")
    print(f"每组重复次数: {n_repeat}")
    print("=" * 60)

    for value in param_values:
        print(f"\n=== 当前参数 {param_name} = {value} ===")
        all_records = []

        for i in range(n_repeat):
            params = fixed_params.copy()
            params[param_name] = value

            print(f"  ▶️ 实验 {i + 1}/{n_repeat} 运行中...\n", end=" ")
            try:
                result = dqn_optimize(**params)
                record = {
                    "run_index": i + 1,
                    "time_steps": result.get("time_steps", np.nan),
                    "avg_task_reduction": result.get("avg_task_reduction", np.nan),
                    "convergence_episode": result.get("convergence_episode", np.nan),
                    "total_training_time": result.get("total_training_time", np.nan)
                }

                # 若存在传输速率列表，计算平均值
                if "transmission_rates" in result:
                    rates = np.array(result["transmission_rates"], dtype=float)
                    record["avg_transmission_rate"] = np.mean(rates) if len(rates) > 0 else np.nan

                all_records.append(record)
                print("✅ 成功")
            except Exception as e:
                print(f"❌ 失败: {e}")
                all_records.append({
                    "run_index": i + 1,
                    "time_steps": np.nan,
                    "avg_task_reduction": np.nan,
                    "convergence_episode": np.nan,
                    "total_training_time": np.nan,
                    "avg_transmission_rate": np.nan
                })

        # 转为 DataFrame
        df = pd.DataFrame(all_records)

        # 保存 CSV
        csv_name = f"{method_name}_result_{param_name}_{value}.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"📄 已保存结果: {csv_path}")

        # 计算平均值和标准差
        metrics = ["time_steps", "avg_task_reduction", "convergence_episode", "total_training_time"]
        print(f"\n📊 参数 {param_name} = {value} 的统计结果:")
        print("-" * 60)
        for m in metrics:
            mean = df[m].mean(skipna=True)
            std = df[m].std(skipna=True)
            print(f"{m:<22} 均值: {mean:10.4f} | 标准差: {std:10.4f}")
        print("-" * 60)

    print("\n✅ 所有实验完成！结果保存在 ./results/ 文件夹内。")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    run_dso_experiments()
