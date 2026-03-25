import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    """A2C网络：同时输出策略（动作分布）和状态值"""

    def __init__(self, state_size, action_size, num_users, v_max=0.2):
        super(ActorCritic, self).__init__()
        # 输入：基站位置(2) + 用户相对位置(num_users*2) + 剩余任务(num_users)
        input_dim = 2 + num_users * 2 + num_users

        # 共享的特征提取层
        self.fc_shared1 = nn.Linear(input_dim, 64)
        self.fc_shared2 = nn.Linear(64, 64)
        self.fc_shared3 = nn.Linear(64, 32)

        # 演员网络（策略网络）
        self.fc_actor = nn.Linear(32, action_size)

        # 评论家网络（价值网络）
        self.fc_critic = nn.Linear(32, 1)

        # 速度上限
        self.v_max = v_max
        self.action_size = action_size

        # 初始化权重
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m in [self.fc_actor, self.fc_critic]:
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0.0)
            else:
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # 共享特征提取
        x = F.relu(self.fc_shared1(x))
        x = F.relu(self.fc_shared2(x))
        x = F.relu(self.fc_shared3(x))

        # 演员网络输出动作概率分布
        action_probs = F.softmax(self.fc_actor(x), dim=-1)

        # 评论家网络输出状态价值
        state_value = self.fc_critic(x)

        return action_probs, state_value


class A2CAgent:
    """A2C智能体"""

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

        # A2C网络
        self.model = ActorCritic(state_size, action_size, num_users, v_max=max_step).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.entropy_coef = 0.01  # 熵正则化系数

    def select_action(self, state, eval_mode=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if eval_mode:
            with torch.no_grad():
                action_probs, state_value = self.model(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
        else:
            action_probs, state_value = self.model(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

        return action.item(), log_prob, state_value

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
        # 将动作转换为移动
        angle = action * (2 * math.pi / self.action_size)
        dx = torch.cos(torch.tensor(angle)) * self.max_step
        dy = torch.sin(torch.tensor(angle)) * self.max_step
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

    def update(self, rewards, log_probs, state_values, dones):
        """更新网络参数"""
        # 确保所有张量都在正确的设备上
        log_probs = torch.stack(log_probs).to(self.device)
        state_values = torch.stack(state_values).squeeze().to(self.device)

        returns = []
        R = 0
        # 计算回报
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # 计算优势函数
        advantages = returns - state_values.detach()

        # 计算演员损失（策略梯度）
        actor_loss = -(log_probs * advantages).mean()

        # 计算评论家损失（价值函数误差）
        critic_loss = F.mse_loss(state_values, returns)

        # 计算熵正则化
        entropy = -(log_probs * torch.exp(log_probs)).mean()

        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()


def initialize_environment(num_users, area_size):
    """初始化环境：用户位置和任务量"""
    # 用户随机分布在[-area_size/2, area_size/2]的正方形区域内
    user_positions = (torch.rand(num_users, 2) - 0.5) * area_size
    user_positions = user_positions.to(device)  # 移动到设备
    # 每个用户的任务量在[0.5,1]之间随机（避免任务太小）
    user_tasks = torch.rand(num_users) * 2 + 2
    user_tasks = user_tasks.to(device)  # 移动到设备
    # 基站初始化在环境中心
    base_station_pos = torch.zeros((2,))
    base_station_pos = base_station_pos.to(device)  # 移动到设备
    return user_positions, user_tasks, base_station_pos


def a2c_optimize(num_users, area_size, h_unit, sigma, max_step, time_step_duration,
                 convergence_threshold=1e-3, max_episodes=1000):
    """执行A2C优化并返回结果指标"""
    # 初始化环境
    user_positions, initial_tasks, base_station_pos = initialize_environment(num_users, area_size)
    # 将h_unit和sigma也移动到设备
    h_unit = torch.tensor(h_unit)
    sigma = torch.tensor(sigma)
    h_unit = h_unit.to(device)
    sigma = sigma.to(device)
    # 状态和动作空间大小
    state_size = 2 + num_users * 2 + num_users  # 无人机位置(2) + 相对位置(2*num_users) + 剩余任务(num_users)
    action_size = 8  # 8个方向

    # 创建A2C智能体
    agent = A2CAgent(state_size, action_size, num_users, area_size, h_unit, sigma, max_step, time_step_duration)

    # 训练参数
    max_steps_per_episode = 1000

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

        # 存储回合数据
        log_probs = []
        values = []
        rewards = []
        dones = []

        episode_reward = 0

        for step in range(max_steps_per_episode):
            # 选择动作
            action, log_prob, value = agent.select_action(state.cpu().numpy() if state.is_cuda else state)

            # 执行动作
            next_base_pos, next_remaining_tasks, reward, next_state, done = agent.step(
                base_pos, action, user_positions, remaining_tasks)

            # 存储数据
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            # 更新状态
            state = next_state
            base_pos = next_base_pos
            remaining_tasks = next_remaining_tasks
            episode_reward += reward

            # 检查是否结束
            if done:
                break

        # 更新网络
        if len(rewards) > 0:  # 确保有数据可以更新
            loss = agent.update(rewards, log_probs, values, dones)
            losses.append(loss)

        # 记录奖励
        episode_rewards.append(episode_reward)

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
            current_loss = losses[-1] if losses else 0
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Loss: {current_loss:.4f}")

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
    agent.model.eval()  # 设置为评估模式
    base_pos = base_station_pos.clone()
    remaining_tasks = initial_tasks.clone()
    state = agent.get_state(base_pos, user_positions, remaining_tasks)

    time_steps = 0
    transmission_rates = []

    for step in range(max_steps_per_episode):
        # 选择动作（评估模式）
        action, _, _ = agent.select_action(state.cpu().numpy() if state.is_cuda else state, eval_mode=True)

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