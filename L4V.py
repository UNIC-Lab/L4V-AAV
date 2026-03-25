import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# 环境参数
num_users = 4  # 用户数量
area_size = 10.0  # 环境区域大小
h_unit = 1.0  # 单位距离的信道增益
sigma = 0.1  # 噪声功率
unit_move = 0.2  # 基站每步移动距离（降低以提高控制精度）
time_step = 0.1  # 每个时间步长

# 速度上限（沿用原先 unit_move 的量级，作为速度的上界）
v_max = unit_move
# 角度预激活 L2 正则权重（可根据稳定性调整 1e-4 ~ 1e-2）
lambda_preact = 1e-3


def initialize_environment(num_users, area_size):
    """初始化环境：用户位置和任务量"""
    # 用户随机分布在[-area_size/2, area_size/2]的正方形区域内
    user_positions = (torch.rand((num_users, 2)) - 0.5) * area_size

    # 每个用户的任务量在[0.5,1]之间随机（避免任务太小）
    user_tasks = torch.rand(num_users) * 2 + 2

    # 基站初始化在环境中心
    base_station_pos = torch.zeros((1, 2))

    return user_positions, user_tasks, base_station_pos


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
    rates = compute_transmission_rate(base_pos, user_positions, h_unit, sigma) * 1

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


# 初始化环境
user_positions, initial_tasks, base_station_pos = initialize_environment(num_users, area_size)

# 初始化模型和优化器
model = PathPlanningNet(num_users, v_max=v_max)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

# 训练参数
num_episodes = 1000 + 1
trajectory_length = 100

# 训练记录
loss_history = []
final_remaining_tasks_history = []
all_angles = []
all_speeds = []

for episode in range(num_episodes):
    model.train()
    user_positions, initial_tasks, base_station_pos = initialize_environment(num_users, area_size)
    # 重置环境
    base_pos = base_station_pos.clone()
    remaining_tasks = initial_tasks.clone()

    total_loss = 0.0
    l2_preact_reg = 0.0

    angles_seq = []
    speeds_seq = []

    # —— 单图端到端 roll-out —— #
    for t in range(trajectory_length):
        # 网络同时给出角度与速度，以及角度预激活
        angle, speed, angle_preact = model(base_pos, user_positions, remaining_tasks)

        # 累计角度预激活 L2 正则（用 mean 更稳定）
        #l2_preact_reg = l2_preact_reg + torch.mean(angle_preact ** 2)

        # 状态转移（由角度+速度决定）
        base_pos = g(base_pos, angle, speed)

        # 步进损失（剩余任务 + 位置约束），并前推 remaining_tasks
        step_loss, remaining_tasks = f(base_pos, user_positions, remaining_tasks,
                                       h_unit, sigma, time_step, area_size)
        total_loss = total_loss + step_loss / torch.sum(initial_tasks)

        angles_seq.append(angle)
        speeds_seq.append(speed)

        # 提前结束
        if torch.sum(remaining_tasks) < 0.001 * num_users:
            break

    # 平滑正则（可选，与原逻辑对齐）
    if len(angles_seq) > 1:
        angles_tensor = torch.stack(angles_seq).squeeze()  # [T]
        angle_diff = torch.diff(angles_tensor)
        smoothness_penalty = 0.01 * torch.mean(angle_diff ** 2)
        total_loss = total_loss + smoothness_penalty

    # —— 角度预激活 L2 正则注入 —— #
    total_loss = total_loss + lambda_preact * l2_preact_reg

    # 反向传播与更新
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # 记录
    loss_history.append(float(total_loss.item()))
    final_remaining = float(torch.sum(remaining_tasks).item())
    final_remaining_tasks_history.append(final_remaining)

    # 采样角度/速度分布用于直方图
    all_angles.extend([float(a.item()) for a in angles_seq])
    all_speeds.extend([float(s.item()) for s in speeds_seq])

    # 打印
    if episode % 100 == 0:
        avg_angle = float(torch.mean(torch.stack(angles_seq)).item()) if angles_seq else 0.0
        avg_speed = float(torch.mean(torch.stack(speeds_seq)).item()) if speeds_seq else 0.0
        print(f"Episode {episode}, Loss: {total_loss.item():.4f}, "
              f"Final remaining: {final_remaining:.4f}, "
              f"Avg angle: {avg_angle:.4f}, Avg speed: {avg_speed:.4f}")

# —— 最终轨迹可视化 —— #
plt.figure(figsize=(10, 8))

# 损失曲线
plt.subplot(2, 2, 1)
plt.plot(loss_history)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# 最终剩余
plt.subplot(2, 2, 2)
plt.plot(final_remaining_tasks_history)
plt.xlabel('Episode')
plt.ylabel('Final Remaining')
plt.title('Final Remaining Tasks Over Training')
plt.grid(True)

# 角度分布
plt.subplot(2, 2, 3)
plt.hist(all_angles, bins=20, alpha=0.7)
plt.xlabel('Angle (radians)')
plt.ylabel('Frequency')
plt.title('Angle Distribution')
plt.grid(True)

# 速度分布
plt.subplot(2, 2, 4)
plt.hist(all_speeds, bins=20, alpha=0.7)
plt.xlabel('Speed')
plt.ylabel('Frequency')
plt.title('Speed Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

# —— 用训练后策略跑一次轨迹 —— #
base_pos = base_station_pos.clone()
remaining_tasks = initial_tasks.clone()
trajectory = [base_pos.clone().detach().numpy()]

# —— 用训练后策略跑一次轨迹，并记录角度/速度 —— #
with torch.no_grad():
    base_pos = base_station_pos.clone()
    remaining_tasks = initial_tasks.clone()

    traj_xy = []  # [(x,y), ...]
    traj_angles = []  # [angle_t, ...]
    traj_speeds = []  # [speed_t, ...]

    for t in range(trajectory_length):
        angle, speed, _ = model(base_pos, user_positions, remaining_tasks)
        # 记录当前位姿（注意 base_pos 形状可能是 [1,2]）
        traj_xy.append(base_pos.view(-1).detach().cpu().numpy())
        traj_angles.append(float(angle.item()))
        traj_speeds.append(float(speed.item()))

        # 状态推进
        base_pos = g(base_pos, angle, speed)

        # 任务推进
        rates = compute_transmission_rate(base_pos, user_positions, h_unit, sigma)
        transmitted = rates * time_step
        remaining_tasks = torch.clamp(remaining_tasks - transmitted, min=0)
        if torch.sum(remaining_tasks) < 0.001 * num_users:
            break

    # 记录最终位置
    traj_xy.append(base_pos.view(-1).detach().cpu().numpy())

traj_xy = np.asarray(traj_xy)  # 形状 [T+1, 2]

# —— 最后一次运行的轨迹可视化（修正版） —— #
plt.figure(figsize=(7, 7))

# UAV 连续路径
plt.plot(traj_xy[:, 0], traj_xy[:, 1], '-', linewidth=2, alpha=0.9, label='UAV path')

# 方向箭头（抽样画，避免太密）
dirs = np.diff(traj_xy, axis=0)
if len(dirs) > 0:
    step = max(1, len(dirs) // 25)  # 最多画 25 个箭头
    plt.quiver(traj_xy[:-1:step, 0], traj_xy[:-1:step, 1],
               dirs[::step, 0], dirs[::step, 1],
               angles='xy', scale_units='xy', scale=1.0, width=0.003, alpha=0.6, label='Direction')

# 起点/终点
plt.scatter(traj_xy[0, 0], traj_xy[0, 1], s=80, marker='o', edgecolor='k', label='Start')
plt.scatter(traj_xy[-1, 0], traj_xy[-1, 1], s=100, marker='X', edgecolor='k', label='End')

# 用户位置
up = user_positions.detach().cpu().numpy()
tasks0 = initial_tasks.detach().cpu().float().numpy()  # 初始任务（评估时未被消耗）
tmin, tmax = float(tasks0.min()), float(tasks0.max())
s_min, s_max = 20.0, 220.0  # 可调：最小/最大面积（points^2）
eps = 1e-8

if (tmax - tmin) < eps:
    sizes = np.full_like(tasks0, (s_min + s_max) / 2.0)
else:
    sizes = s_min + (tasks0 - tmin) / (tmax - tmin) * (s_max - s_min)

plt.scatter(up[:, 0], up[:, 1],
            s=sizes, alpha=0.65, edgecolor='k', linewidths=0.3,
            label='Users (size ∝ initial tasks)')

# ====== 坐标范围设定：二选一 ======

# 【方式 A：对称域（推荐）】
half = area_size / 2.0
plt.xlim(-half, half)
plt.ylim(-half, half)

# # 【方式 B：数据自适应】
# all_pts = np.vstack([traj_xy, up])
# xmin, ymin = all_pts.min(axis=0)
# xmax, ymax = all_pts.max(axis=0)
# pad_x = max(1e-3, 0.05 * (xmax - xmin))
# pad_y = max(1e-3, 0.05 * (ymax - ymin))
# plt.xlim(xmin - pad_x, xmax + pad_x)
# plt.ylim(ymin - pad_y, ymax + pad_y)

# 画出区域边界框（可选，更直观地呈现对称域）
rect = plt.Rectangle((-half, -half), area_size, area_size,
                     fill=False, linestyle='--', linewidth=1.0, alpha=0.5, label='Boundary')
plt.gca().add_patch(rect)

# 原点十字线（可选，便于判断象限）
plt.axhline(0.0, linewidth=0.8, alpha=0.3)
plt.axvline(0.0, linewidth=0.8, alpha=0.3)

# 其他绘图设置
plt.gca().set_aspect('equal', 'box')
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
plt.title('Final Rollout Trajectory (angle+speed)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
