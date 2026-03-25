import numpy as np
import math
import random
import time
from deap import base, creator, tools, algorithms

# 检查是否已经存在这些类，如果不存在则创建
if not hasattr(creator, 'FitnessMin'):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def initialize_environment(num_users, area_size):
    """初始化环境：用户位置和任务量"""
    # 用户随机分布在[-area_size/2, area_size/2]的正方形区域内
    user_positions = (np.random.rand(num_users, 2) - 0.5) * area_size

    # 每个用户的任务量在[0.5,1]之间随机（避免任务太小）
    user_tasks = np.random.rand(num_users) * 2 + 2

    return user_positions, user_tasks


def gen_random_vector(max_step_size):
    """生成随机位移向量（模长不超过max_step_size）"""
    angle = random.uniform(0, 2 * math.pi)
    length = random.uniform(0, max_step_size)
    dx = length * math.cos(angle)
    dy = length * math.sin(angle)
    return (dx, dy)


def create_smart_individual(num_time_steps, user_positions, user_tasks, max_step_size):
    """创建更智能的初始个体，偏向飞向用户"""
    individual = []

    # 策略：大部分时间步飞向任务量最大的用户
    target_user = np.argmax(np.array(user_tasks))
    target_x, target_y = user_positions[target_user]

    # 计算总位移
    total_dx = target_x
    total_dy = target_y

    # 平均分配到各个时间步，加上随机扰动
    for _ in range(num_time_steps):
        dx = total_dx / num_time_steps + random.uniform(-0.05, 0.05)
        dy = total_dy / num_time_steps + random.uniform(-0.05, 0.05)

        # 确保位移不超过最大限制
        mag = math.sqrt(dx ** 2 + dy ** 2)
        if mag > max_step_size:
            scale = max_step_size / mag
            dx *= scale
            dy *= scale

        individual.append((dx, dy))

    return individual


def repair_individual(individual, max_step_size):
    """修复个体：确保每个位移向量模长不超过max_step_size"""
    for i in range(len(individual)):
        dx, dy = individual[i]
        mag = math.sqrt(dx ** 2 + dy ** 2)
        if mag > max_step_size:
            scale = max_step_size / mag
            individual[i] = (dx * scale, dy * scale)
    return individual


def eval_fitness_improved(individual, user_positions, user_tasks, h0, sigma, time_step_duration, num_time_steps):
    """改进的适应度函数：添加惩罚项"""
    # 计算轨迹位置（从(0,0)开始）
    positions = []
    current_x, current_y = 0.0, 0.0
    positions.append((current_x, current_y))
    for dx, dy in individual:
        current_x += dx
        current_y += dy
        positions.append((current_x, current_y))

    # 初始化剩余数据量
    remaining_data = user_tasks.copy()
    total_time_steps_needed = num_time_steps  # 默认最大时间步

    # 模拟每个时间步的数据传输
    for t in range(num_time_steps):
        pos = positions[t]  # 时间步t的无人机位置
        for i in range(len(user_positions)):
            if remaining_data[i] <= 0:
                continue
            user_pos = user_positions[i]
            distance = math.sqrt((pos[0] - user_pos[0]) ** 2 + (pos[1] - user_pos[1]) ** 2)

            # 避免除零错误
            if distance < 0.001:
                distance = 0.001

            h = h0 / (distance ** 2)
            rate = math.log2(1 + h)  # 使用log2计算速率
            data_transmitted = rate * time_step_duration

            if data_transmitted > remaining_data[i]:
                data_transmitted = remaining_data[i]
            remaining_data[i] -= data_transmitted

        # 检查所有用户是否完成传输
        if all(d <= 0.001 * len(user_positions) for d in remaining_data):
            total_time_steps_needed = t + 1  # 完成于时间步t
            break

    # 如果没有完成所有传输，添加惩罚项
    penalty = 0
    unfinished_data = sum(max(0, d) for d in remaining_data)
    if unfinished_data > 0.001 * len(user_positions):
        # 惩罚与剩余数据量成正比，乘以一个大系数
        penalty = unfinished_data * 20000

    total_time = total_time_steps_needed ** 2 + penalty

    return total_time,


def cxTwoPointVector(ind1, ind2, max_step_size):
    """自定义交叉操作（两点交叉）"""
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    # 修复交叉后的个体
    repair_individual(ind1, max_step_size)
    repair_individual(ind2, max_step_size)

    return ind1, ind2


def mutate_individual_improved(individual, indpb, max_step_size):
    """改进的变异操作"""
    for i in range(len(individual)):
        if random.random() < indpb:
            # 多种变异策略
            if random.random() < 0.8:  # 80%概率小幅扰动
                dx, dy = individual[i]
                angle_change = random.uniform(-0.4, 0.4)  # 小幅改变角度
                length_change = random.uniform(-0.08, 0.08) * max_step_size  # 小幅改变长度


                angle = math.atan2(dy, dx) + angle_change
                length = math.sqrt(dx ** 2 + dy ** 2) + length_change
                length = max(0, min(length, max_step_size))  # 确保在范围内

                individual[i] = (length * math.cos(angle), length * math.sin(angle))
            else:  # 20%概率完全随机
                individual[i] = gen_random_vector(max_step_size)
    return individual,


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                        halloffame=None, verbose=__debug__):
    """带有精英保留的进化算法"""
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 评估初始种群
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # 开始进化
    for gen in range(1, ngen + 1):
        # 选择下一代
        offspring = toolbox.select(population, len(population))
        mutpb = 0.3 * (1 - gen / ngen)
        # 变异和交叉
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # 评估新个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 更新Hall of Fame
        if halloffame is not None:
            halloffame.update(offspring)

        # 精英保留：用Hall of Fame中的最佳个体替换最差个体
        if halloffame is not None and len(halloffame) > 0:
            # 找到最差的个体
            offspring.sort(key=lambda ind: ind.fitness.values)
            # 用精英替换最差的个体
            offspring[0] = toolbox.clone(halloffame[0])

        # 替换种群
        population[:] = offspring

        # 记录统计信息
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def analyze_solution(individual, user_positions, user_tasks, h0, sigma, time_step_duration, num_time_steps):
    """分析解决方案的性能"""
    positions = []
    current_x, current_y = 0.0, 0.0
    positions.append((current_x, current_y))
    for dx, dy in individual:
        current_x += dx
        current_y += dy
        positions.append((current_x, current_y))

    # 初始化剩余数据量
    remaining_data = user_tasks.copy()
    completion_time = num_time_steps

    # 记录每个时间步的传输情况
    transmission_log = []
    transmission_rates = []

    # 模拟每个时间步的数据传输
    for t in range(num_time_steps):
        pos = positions[t]
        step_transmission = []
        step_rates = []

        for i in range(len(user_positions)):
            if remaining_data[i] <= 0.001 * len(user_positions):
                step_transmission.append(0)
                step_rates.append(0)
                continue

            user_pos = user_positions[i]
            distance = math.sqrt((pos[0] - user_pos[0]) ** 2 + (pos[1] - user_pos[1]) ** 2)

            if distance < 0.001:
                distance = 0.001

            h = h0 / (distance ** 2)
            rate = math.log2(1 + h / sigma)
            data_transmitted = rate * time_step_duration

            if data_transmitted > remaining_data[i]:
                data_transmitted = remaining_data[i]
            remaining_data[i] -= data_transmitted
            step_transmission.append(data_transmitted)
            step_rates.append(rate)

        transmission_log.append(step_transmission)
        transmission_rates.extend(step_rates)

        # 检查是否完成
        if all(d <= 0.001 * len(user_positions) for d in remaining_data):
            completion_time = t + 1
            break

    return completion_time, transmission_rates


def ga_optimize(num_users, area_size, h_unit, sigma, max_step, time_step_duration,
                convergence_threshold=1e-3, max_generations=5000):
    """执行GA优化并返回结果指标"""
    # 初始化环境
    user_positions, user_tasks = initialize_environment(num_users, area_size)

    # 遗传算法参数
    num_time_steps = 500
    population_size = 300
    crossover_prob = 0.7
    mutation_prob = 0.3

    # 初始化工具
    toolbox = base.Toolbox()

    # 注册个体和种群创建函数
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: create_smart_individual(num_time_steps, user_positions, user_tasks, max_step))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册评估函数
    toolbox.register("evaluate", eval_fitness_improved,
                     user_positions=user_positions, user_tasks=user_tasks,
                     h0=h_unit, sigma=sigma, time_step_duration=time_step_duration,
                     num_time_steps=num_time_steps)

    # 注册遗传操作
    toolbox.register("mate", cxTwoPointVector, max_step_size=max_step)
    toolbox.register("mutate", mutate_individual_improved, indpb=0.1, max_step_size=max_step)
    toolbox.register("select", tool
    s.selTournament, tournsize=5)

    # 创建初始种群
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)  # 保存最佳个体
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 记录训练过程
    start_time = time.time()

    # 运行遗传算法
    pop, log = eaSimpleWithElitism(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
                                   ngen=max_generations, stats=stats, halloffame=hof, verbose=False)

    # 计算总训练时间
    total_training_time = time.time() - start_time

    # 获取最佳个体
    best_individual = hof[0]

    # 分析最佳解的性能
    completion_time, transmission_rates = analyze_solution(
        best_individual, user_positions, user_tasks, h_unit, sigma, time_step_duration, num_time_steps)

    # 计算平均每步减少的任务量
    total_tasks = np.sum(user_tasks)
    avg_task_reduction = total_tasks / completion_time if completion_time > 0 else 0

    # GA没有明确的收敛轮次概念，使用找到最佳解时的代数
    # 这里我们简单使用总代数作为收敛轮次
    convergence_generation = max_generations

    return {
        'time_steps': completion_time,
        'convergence_episode': convergence_generation,
        'convergence_time': total_training_time,
        'total_training_time': total_training_time,
        'avg_task_reduction': avg_task_reduction,
        'transmission_rates': transmission_rates
    }