import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import math

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Actor(nn.Module):
    """DDPG演员网络：输出连续动作（角度和速度）"""

    def __init__(self, state_size, action_size, num_users, v_max=0.2):
        super(Actor, self).__init__()
        # 输入：基站位置(2) + 用户相对位置(num_users*2) + 剩余任务(num_users)
        input_dim = 2 + num_users * 2 + num_users

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)

        # 输出层：角度（tanh→[-1, 1]映射到[-π, π]）和速度（sigmoid→[0, v_max]）
        self.angle_head = nn.Linear(32, 1)
        self.speed_head = nn.Linear(32, 1)

        # 速度上限
        self.v_max = v_max

        # 初始化权重
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m in [self.angle_head, self.speed_head]:
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0.0)
            else:
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # 角度输出（tanh→[-1, 1]映射到[-π, π]）
        angle = torch.tanh(self.angle_head(x)) * math.pi

        # 速度输出（sigmoid→[0, v_max]）
        speed = torch.sigmoid(self.speed_head(x)) * self.v_max

        return torch.cat([angle, speed], dim=-1)


class Critic(nn.Module):
    """DDPG评论家网络：评估状态-动作对的价值"""

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        # 输入：状态 + 动作
        self.fc1 = nn.Linear(state_size + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # 初始化权重
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
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


class DDPGAgent:
    """DDPG智能体"""

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

        # 演员和评论家网络
        self.actor = Actor(state_size, action_size, num_users, v_max=max_step).to(self.device)
        self.actor_target = Actor(state_size, action_size, num_users, v_max=max_step).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)

        # 经验回放
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64

        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005  # 目标网络软更新参数
        self.noise_std = 0.2  # 探索噪声标准差
        self.noise_decay = 0.995  # 噪声衰减率
        self.min_noise_std = 0.01  # 最小噪声标准差

        # 训练步骤计数器
        self.steps_done = 0

    def select_action(self, state, add_noise=True):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        # 添加探索噪声
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            action = action + noise
            # 确保动作在有效范围内
            action[0] = np.clip(action[0], -math.pi, math.pi)  # 角度限制在[-π, π]
            action[1] = np.clip(action[1], 0, self.max_step)  # 速度限制在[0, max_step]

        return action

    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.batch_size:
            return 0, 0

        # 从内存中采样
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.FloatTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # 更新评论家网络
        with torch.no_grad():
            next_actions = self.actor_target(next_state_batch)
            next_q_values = self.critic_target(next_state_batch, next_actions)
            target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        current_q_values = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 更新演员网络
        actor_actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 衰减探索噪声
        self.noise_std = max(self.min_noise_std, self.noise_std * self.noise_decay)

        # 更新步骤计数器
        self.steps_done += 1

        return critic_loss.item(), actor_loss.item()

    def compute_transmission_rate(self, base_pos, user_positions):
        """计算传输速率"""
        # 计算距离的平方
        distances_squared = torch.sum((base_pos - user_positions) ** 2, dim=1)

        # 避免除零，设置最小距离
        distances_squared = torch.clamp(distances_squared, 0.1, float('inf'))

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
        return state

    def step(self, base_pos, action, user_positions, remaining_tasks):
        """执行一步动作"""
        # 解析动作（角度和速度）
        angle, speed = action

        # 将动作转换为移动
        dx = torch.cos(torch.tensor(angle)) * speed
        dy = torch.sin(torch.tensor(angle)) * speed
        movement = torch.tensor([dx, dy]).to(self.device)

        # 更新无人机位置
        new_base_pos = base_pos + movement

        # 计算传输速率
        rates = self.compute_transmission_rate(new_base_pos, user_positions)

        # 计算传输量
        transmitted = rates * self.time_step_duration

        # 更新剩余任务量
        new_remaining_tasks = torch.clamp(remaining_tasks - transmitted, 0, float('inf'))

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
    user_positions = (torch.rand((num_users, 2)) - 0.5) * area_size
    user_positions = user_positions.to(device)  # 移动到设备

    # 每个用户的任务量在[0.5,1]之间随机（避免任务太小）
    user_tasks = torch.rand(num_users) * 2 + 2
    user_tasks = user_tasks.to(device)  # 移动到设备

    # 基站初始化在环境中心
    base_station_pos = torch.zeros((2,))
    base_station_pos = base_station_pos.to(device)  # 移动到设备

    return user_positions, user_tasks, base_station_pos


def ddpg_optimize(num_users, area_size, h_unit, sigma, max_step, time_step_duration,
                  convergence_threshold=1e-3, max_episodes=2000):
    """执行DDPG优化并返回结果指标"""
    # 初始化环境
    user_positions, initial_tasks, base_station_pos = initialize_environment(num_users, area_size)
    # 将h_unit和sigma也移动到设备
    h_unit = torch.tensor(h_unit)
    sigma = torch.tensor(sigma)
    h_unit = h_unit.to(device)
    sigma = sigma.to(device)

    # 状态和动作空间大小
    state_size = 2 + num_users * 2 + num_users  # 无人机位置(2) + 相对位置(2*num_users) + 剩余任务(num_users)
    action_size = 2  # 角度和速度

    # 创建DDPG智能体
    agent = DDPGAgent(state_size, action_size, num_users, area_size, h_unit, sigma, max_step, time_step_duration)

    # 训练参数
    max_steps_per_episode = 1000

    # 记录训练过程
    episode_rewards = []
    critic_losses = []
    actor_losses = []
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
        episode_critic_loss = 0
        episode_actor_loss = 0
        update_count = 0

        for step in range(max_steps_per_episode):
            # 选择动作
            action = agent.select_action(state.cpu().numpy() if state.is_cuda else state, add_noise=True)

            # 执行动作
            next_base_pos, next_remaining_tasks, reward, next_state, done = agent.step(
                base_pos, action, user_positions, remaining_tasks)

            # 存储经验
            agent.memory.push(state.cpu().numpy() if state.is_cuda else state, action, reward,
                              next_state.cpu().numpy() if next_state.is_cuda else next_state, done)

            # 优化模型
            critic_loss, actor_loss = agent.optimize_model()
            if critic_loss != 0:
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                update_count += 1

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

        # 记录平均损失
        if update_count > 0:
            critic_losses.append(episode_critic_loss / update_count)
            actor_losses.append(episode_actor_loss / update_count)

        # 检查收敛条件
        if len(episode_rewards) >= 10:
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
            avg_critic_loss = torch.mean(torch.tensor(critic_losses[-10:])).item() if critic_losses else 0
            avg_actor_loss = torch.mean(torch.tensor(actor_losses[-10:])).item() if actor_losses else 0
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Noise: {agent.noise_std:.3f}, "
                  f"Critic Loss: {avg_critic_loss:.4f}, Actor Loss: {avg_actor_loss:.4f}")

        # 提前停止条件
        if episode > 100 and torch.mean(torch.tensor(episode_rewards[-10:])).item() > 2000:
            break

    end = torch.cuda.Event(enable_timing=True)
    end.record()

    # 等待 CUDA 操作完成
    torch.cuda.synchronize()
    # 计算最终性能指标
    total_training_time = start.elapsed_time(end) / 1000
    # 评估训练后的模型
    base_pos = base_station_pos.clone()
    remaining_tasks = initial_tasks.clone()
    state = agent.get_state(base_pos, user_positions, remaining_tasks)

    time_steps = 0
    transmission_rates = []

    for step in range(max_steps_per_episode):
        # 选择动作（评估模式，不添加噪声）
        action = agent.select_action(state.cpu().numpy() if state.is_cuda else state, add_noise=False)

        # 执行动作
        next_base_pos, next_remaining_tasks, reward, next_state, done = agent.step(
            base_pos, action, user_positions, remaining_tasks)

        # 记录传输速率
        rates = agent.compute_transmission_rate(next_base_pos, user_positions)
        transmission_rates.extend(rates.tolist())

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

    # 如果未收敛，使用最大训练轮次
    if convergence_episode is None:
        convergence_episode = episode
        convergence_time = total_training_time

    return {
        'time_steps': time_steps,
        'convergence_episode': convergence_episode,
        'convergence_time': convergence_time,
        'total_training_time': total_training_time,
        'avg_task_reduction': avg_task_reduction,
        'transmission_rates': transmission_rates
    }