import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PathPlanningNet(nn.Module):
    """路径规划神经网络：同时输出 angle 与 speed，并返回 angle_preact 以便正则化"""

    def __init__(self, num_users, v_max=0.2):
        super(PathPlanningNet, self).__init__()
        # 输入：基站位置(2) + 用户相对位置(num_users*2) + 剩余任务(num_users)
        input_dim = 2 + num_users * 2 + num_users

        self.fc_1 = nn.Linear(input_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, 32)

        # 两个输出头：角度（tanh→[-π, π]），速度（sigmoid→[0, v_max]）
        self.angle_head = nn.Linear(32, 1)
        self.speed_head = nn.Linear(32, 1)

        # 速度上限
        self.v_max = v_max

        # 初始化
        nn.init.xavier_uniform_(self.fc_1.weight)
        nn.init.xavier_uniform_(self.fc_2.weight)
        nn.init.xavier_uniform_(self.fc_3.weight)
        nn.init.uniform_(self.angle_head.weight, -0.01, 0.01)
        nn.init.uniform_(self.speed_head.weight, -0.01, 0.01)
        self.device = device

    def forward(self, base_pos, user_positions, remaining_tasks):
        # 相对位置作为输入更稳定
        relative_positions = user_positions - base_pos

        # 拼接输入向量
        x = torch.cat([base_pos.flatten(),
                       relative_positions.flatten(),
                       remaining_tasks], dim=0)

        # 主干网络
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        z = F.relu(self.fc_3(x))  # 共享特征

        # —— 角度分支：预激活 + tanh 压到 [-π, π] ——
        angle_preact = self.angle_head(z)  # 形状 [1]
        angle = torch.tanh(angle_preact) * torch.pi  # [-π, π]

        # —— 速度分支：sigmoid 压到 [0, v_max]（可微、稳定） ——
        speed_preact = self.speed_head(z)  # 形状 [1]
        speed = torch.sigmoid(speed_preact) * self.v_max  # [0, v_max]

        # 返回 angle, speed, 以及 angle_preact 以便正则
        return angle, speed, angle_preact


def compute_transmission_rate(base_pos, user_positions, h_unit, sigma):
    """计算基站与各用户之间的传输速率"""
    # 计算距离的平方
    distances_squared = torch.sum((base_pos - user_positions) ** 2, dim=1)

    # 避免除零，设置最小距离
    distances_squared = torch.clamp(distances_squared, min=0.1)

    # 计算信道增益
    h = h_unit / distances_squared

    # 计算传输速率: log(1 + h/sigma)
    rates = torch.log(1 + h / sigma)

    return rates


def f(base_pos, user_positions, remaining_tasks, h_unit, sigma, time_step, area_size):
    """增强的损失函数：剩余任务量 + 位置约束"""
    # 计算当前传输速率
    rates = compute_transmission_rate(base_pos, user_positions, h_unit, sigma)

    # 计算这个时间步内的传输量
    transmitted = rates * time_step

    # 更新剩余任务量（不能小于0）
    new_remaining_tasks = torch.clamp(remaining_tasks - transmitted, min=0)

    # 基础损失：剩余任务总和
    task_loss = torch.sum(new_remaining_tasks)

    # 位置约束：鼓励基站保持在合理范围内
    position_penalty = 0.01 * torch.sum(torch.abs(base_pos) - area_size / 2).clamp(min=0)

    # 总损失
    loss = task_loss + position_penalty

    return loss, new_remaining_tasks


def g(base_pos, angle, speed):
    """状态转移函数：由角度 + 速度决定下一位置（端到端可微）"""
    # angle, speed 形状均为 [1] 或标量；确保拼接维度正确
    dx = (torch.cos(angle) * speed).view(-1, 1)
    dy = (torch.sin(angle) * speed).view(-1, 1)
    movement = torch.cat([dx, dy], dim=-1).view_as(base_pos)
    new_pos = base_pos + movement
    return new_pos


def initialize_environment(num_users, area_size):
    """初始化环境：用户位置和任务量"""
    # 用户随机分布在[-area_size/2, area_size/2]的正方形区域内
    user_positions = (torch.rand((num_users, 2)) - 0.5) * area_size
    user_positions = user_positions.to(device)  # 移动到设备

    # 每个用户的任务量在[0.5,1]之间随机（避免任务太小）
    user_tasks = torch.rand(num_users) * 2 + 2
    user_tasks = user_tasks.to(device)  # 移动到设备

    # 基站初始化在环境中心
    base_station_pos = torch.zeros((1, 2))
    base_station_pos = base_station_pos.to(device)  # 移动到设备

    return user_positions, user_tasks, base_station_pos


def dso_optimize(num_users, area_size, h_unit, sigma, max_step, time_step_duration,
                 convergence_threshold=1e-3, max_episodes=2000):
    """执行DSO优化并返回结果指标"""
    # 初始化环境
    user_positions, initial_tasks, base_station_pos = initialize_environment(num_users, area_size)

    # 将h_unit和sigma也移动到设备
    h_unit = torch.tensor(h_unit)
    sigma = torch.tensor(sigma)
    h_unit = h_unit.to(device)
    sigma = sigma.to(device)

    # 初始化模型和优化器
    model = PathPlanningNet(num_users, v_max=max_step).to(device)  # 模型移动到设备
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

    # 训练参数
    trajectory_length = 500
    lambda_preact = 1e-3

    # 记录训练过程
    loss_history = []
    convergence_episode = None
    prev_loss = float('inf')
    stable_count = 0

    start = torch.cuda.Event(enable_timing=True)
    start.record()
    for episode in range(max_episodes):
        model.train()
        # 重置环境
        base_pos = base_station_pos.clone()
        remaining_tasks = initial_tasks.clone()

        total_loss = 0.0
        l2_preact_reg = 0.0

        angles_seq = []
        speeds_seq = []



        # 单图端到端 roll-out
        for t in range(trajectory_length):





            angle, speed, angle_preact = model(base_pos, user_positions, remaining_tasks)

            # 累计角度预激活 L2 正则
            l2_preact_reg = l2_preact_reg + torch.mean(angle_preact ** 2)

            # 状态转移
            base_pos = g(base_pos, angle, speed)

            # 步进损失
            step_loss, remaining_tasks = f(base_pos, user_positions, remaining_tasks,
                                           h_unit, sigma, time_step_duration, area_size)
            total_loss = total_loss + step_loss / torch.sum(initial_tasks)

            angles_seq.append(angle)
            speeds_seq.append(speed)

            # 提前结束
            if torch.sum(remaining_tasks) < 0.001 * num_users:
                break




        # 平滑正则
        if len(angles_seq) > 1:
            angles_tensor = torch.stack(angles_seq).squeeze()
            angle_diff = torch.diff(angles_tensor)
            smoothness_penalty = 0.01 * torch.mean(angle_diff ** 2)
            total_loss = total_loss + smoothness_penalty

        # 角度预激活 L2 正则注入
        total_loss = total_loss + lambda_preact * l2_preact_reg

        # 反向传播与更新
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录损失
        current_loss = float(total_loss.item())
        loss_history.append(current_loss)

        # 检查收敛条件
        if abs(prev_loss - current_loss) < convergence_threshold:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= 5 and convergence_episode is None:
            convergence_episode = episode - 4  # 连续5次稳定的起始轮次
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            convergence_time = start.elapsed_time(end) / 1000
            break

        prev_loss = current_loss
        # 打印进度
        if episode % 100 == 0:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            print(f"Episode {episode},  Loss: {prev_loss:.4f},TotalTime: {start.elapsed_time(end)/ 1000 :.4f}")

        # 提前停止
        if episode > 100 and current_loss < 0.01:
            break
    end = torch.cuda.Event(enable_timing=True)
    end.record()

    # 等待 CUDA 操作完成
    torch.cuda.synchronize()
    # 计算最终性能指标
    total_training_time = start.elapsed_time(end) / 1000

    # 评估训练后的模型
    with torch.no_grad():
        base_pos = base_station_pos.clone()
        remaining_tasks = initial_tasks.clone()
        time_steps = 0
        transmission_rates = []

        for t in range(trajectory_length):
            angle, speed, _ = model(base_pos, user_positions, remaining_tasks)
            base_pos = g(base_pos, angle, speed)

            rates = compute_transmission_rate(base_pos, user_positions, h_unit, sigma)
            transmitted = rates * time_step_duration
            remaining_tasks = torch.clamp(remaining_tasks - transmitted, min=0)

            transmission_rates.extend(rates.tolist())
            time_steps += 1

            if torch.sum(remaining_tasks) < 0.001 * num_users:
                break

        # 计算平均每步减少的任务量
        total_tasks = torch.sum(initial_tasks).item()
        avg_task_reduction = total_tasks / time_steps if time_steps > 0 else 0

    # 如果未收敛，使用最大训练轮次
    if convergence_episode is None:
        convergence_episode = max_episodes
        convergence_time = total_training_time

    return {
        'time_steps': time_steps,
        'convergence_episode': convergence_episode,
        'convergence_time': convergence_time,
        'total_training_time': total_training_time,
        'avg_task_reduction': avg_task_reduction,
        'transmission_rates': transmission_rates
    }


