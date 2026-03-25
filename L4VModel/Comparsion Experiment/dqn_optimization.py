import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import math
import time

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    """DQN网络"""


    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_size, action_size, num_users, area_size, h_unit, sigma, max_step, time_step_duration):
        self.state_size = state_size
        self.action_size = action_size
        self.num_users = num_users
        self.area_size = area_size
        self.h_unit = h_unit
        self.sigma = sigma
        self.max_step = max_step
        self.time_step_duration = time_step_duration

        # 设备
        self.device = device

        # Q网络和目标网络
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # 经验回放
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64

        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10  # 目标网络更新频率

        # 训练步骤计数器
        self.steps_done = 0

    def select_action(self, state, eval_mode=False):
        """选择动作"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.batch_size:
            return None

        # 从内存中采样
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.stack([torch.FloatTensor(s) for s in batch[0]]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.stack([torch.FloatTensor(s) for s in batch[3]]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)

        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算下一个状态的最大Q值
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # 计算损失
        loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def compute_transmission_rate(self, base_pos, user_positions):
        """计算传输速率"""
        # 计算距离的平方
        distances_squared = torch.sum((base_pos - user_positions) ** 2, dim=1)

        # 避免除零，设置最小距离
        distances_squared = torch.clamp(distances_squared, 0.1, None)

        # 计算信道增益
        h = self.h_unit / distances_squared

        # 计算传输速率: log(1 + h/sigma)
        rates = torch.log(1 + h / self.sigma)

        return rates

    def get_state(self, base_pos, user_positions, remaining_tasks):
        """获取状态表示"""
        # 状态包括：无人机位置、用户相对位置、剩余任务量
        relative_positions = user_positions - base_pos
        state = torch.cat([
            base_pos.flatten(),
            relative_positions.flatten(),
            remaining_tasks
        ])
        return state.cpu()  # 返回CPU上的张量以便存储

    def step(self, base_pos, action, user_positions, remaining_tasks):
        """执行一步动作"""
        # 将动作转换为移动
        angle = action * (2 * math.pi / self.action_size)
        dx = math.cos(angle) * self.max_step
        dy = math.sin(angle) * self.max_step
        movement = torch.tensor([dx, dy], device=self.device)

        # 更新无人机位置
        new_base_pos = base_pos + movement

        # 计算传输速率
        rates = self.compute_transmission_rate(new_base_pos, user_positions)

        # 计算传输量
        transmitted = rates * self.time_step_duration

        # 更新剩余任务量
        new_remaining_tasks = torch.clamp(remaining_tasks - transmitted, 0, None)

        # 计算奖励
        # 奖励与传输的数据量成正比，并鼓励完成任务
        reward = torch.sum(transmitted).item() * 10

        # 如果所有任务完成，给予额外奖励
        if torch.sum(new_remaining_tasks) < 0.001 * self.num_users:
            reward += 100
            done = True
        else:
            done = False

        # 检查是否超出边界
        if torch.any(torch.abs(new_base_pos) > self.area_size / 2):
            reward -= 5  # 超出边界惩罚
            new_base_pos = torch.clamp(new_base_pos, -self.area_size / 2, self.area_size / 2)

        # 获取新状态
        next_state = self.get_state(new_base_pos, user_positions, new_remaining_tasks)

        return new_base_pos, new_remaining_tasks, reward, next_state, done


def initialize_environment(num_users, area_size):
    """初始化环境：用户位置和任务量"""
    # 用户随机分布在[-area_size/2, area_size/2]的正方形区域内
    user_positions = (torch.rand(num_users, 2, device=device) - 0.5) * area_size
    # 每个用户的任务量在[0.5,1]之间随机（避免任务太小）
    user_tasks = torch.rand(num_users, device=device) * 2 + 2
    # 基站初始化在环境中心
    base_station_pos = torch.zeros((2,), device=device)
    return user_positions, user_tasks, base_station_pos


def dqn_optimize(num_users, area_size, h_unit, sigma, max_step, time_step_duration,
                 convergence_threshold=1e-3, max_episodes=2000):
    """执行DQN优化并返回结果指标"""
    # 初始化环境
    user_positions, initial_tasks, base_station_pos = initialize_environment(num_users, area_size)
    h_unit = torch.tensor(h_unit, device=device)
    sigma = torch.tensor(sigma, device=device)

    # 状态和动作空间大小
    state_size = 2 + num_users * 2 + num_users  # 无人机位置(2) + 相对位置(2*num_users) + 剩余任务(num_users)
    action_size = 256  # 2560.个方向

    # 创建DQN智能体
    agent = DQNAgent(state_size, action_size, num_users, area_size, h_unit, sigma, max_step, time_step_duration)

    # 训练参数
    max_steps_per_episode = 500

    # 记录训练过程
    episode_rewards = []

    losses = []
    convergence_episode = None
    prev_avg_reward = float('-inf')
    stable_count = 0

    start = torch.cuda.Event(enable_timing=True)
    start.record()

    for episode in range(max_episodes):
        # 重置环境
        base_pos = base_station_pos.clone()
        remaining_tasks = initial_tasks.clone()
        state = agent.get_state(base_pos, user_positions, remaining_tasks)

        episode_reward = 0
        episode_losses = []

        for step in range(max_steps_per_episode):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_base_pos, next_remaining_tasks, reward, next_state, done = agent.step(
                base_pos, action, user_positions, remaining_tasks)

            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)

            # 优化模型
            loss = agent.optimize_model()
            if loss is not None:
                episode_losses.append(loss)

            # 更新状态
            state = next_state
            base_pos = next_base_pos
            remaining_tasks = next_remaining_tasks
            episode_reward += reward

            # 检查是否结束
            if done:
                break

        # 记录奖励
        episode_rewards.append(episode_reward)

        # 计算平均损失
        avg_loss = torch.mean(torch.tensor(episode_losses)) if episode_losses else 0

        # 检查收敛条件
        if len(episode_rewards) >= 5:
            recent_avg_reward = torch.mean(torch.tensor(episode_rewards[-10:])).item()
            if abs(recent_avg_reward - prev_avg_reward) < convergence_threshold:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= 5 and convergence_episode is None:
                convergence_episode = episode - 4
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                torch.cuda.synchronize()
                convergence_time = start.elapsed_time(end) / 1000
                break

            prev_avg_reward = recent_avg_reward

        # 打印进度
        if episode % 100 == 0:
            print(
                f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Loss: {avg_loss:.4f}")

        # 提前停止条件
        if episode > 100 and torch.mean(torch.tensor(episode_rewards[-10:])).item() > 2000:
            break

    end = torch.cuda.Event(enable_timing=True)
    end.record()

    # 等待 CUDA 操作完成
    torch.cuda.synchronize()
    # 计算最终性能指标
    total_training_time = start.elapsed_time(end) / 1000

    # 计算最终性能指标
    if convergence_episode is None:
        convergence_episode = episode
        convergence_time = total_training_time

    # 评估训练后的模型
    base_pos = base_station_pos.clone()
    remaining_tasks = initial_tasks.clone()
    state = agent.get_state(base_pos, user_positions, remaining_tasks)

    time_steps = 0
    transmission_rates = []

    for step in range(max_steps_per_episode):
        # 选择动作（评估模式，不使用探索）
        action = agent.select_action(state, eval_mode=True)

        # 执行动作
        next_base_pos, next_remaining_tasks, reward, next_state, done = agent.step(
            base_pos, action, user_positions, remaining_tasks)

        # 记录传输速率
        rates = agent.compute_transmission_rate(next_base_pos, user_positions)
        transmission_rates.extend(rates.cpu().tolist())

        # 更新状态
        state = next_state
        base_pos = next_base_pos
        remaining_tasks = next_remaining_tasks
        time_steps += 1

        # 检查是否结束
        if done:
            break

    # 计算平均每步减少的任务量
    total_tasks = torch.sum(initial_tasks).item()
    avg_task_reduction = total_tasks / time_steps if time_steps > 0 else 0

    return {
        'time_steps': time_steps,
        'convergence_episode': convergence_episode,
        'convergence_time': convergence_time,
        'total_training_time': total_training_time,
        'avg_task_reduction': avg_task_reduction,
        'transmission_rates': transmission_rates
    }