import random
import math
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
from object import point, missile, drone, smoke
from main import m1, m2, m3, simulate


# 全局变量，用于multiprocessing
_problem_instance = None


def _evaluate_individual_global(individual):
    """全局评估函数，用于multiprocessing"""
    return _problem_instance.evaluate_individual(individual)


class NSGAII_Problem5:
    """NSGA-II算法求解问题5"""

    def __init__(
        self,
        penalty_factor=0.8,
        min_blocked_threshold=0.01,
        balance_reward=0.3,
        non_overlap_reward=0,  # 降低不重叠奖励
        balance_tolerance=0.4,  # 使用相对容忍度而不是绝对值
    ):
        # 问题参数
        self.missiles = [m1, m2, m3]
        self.drone_positions = [
            point(17800, 0, 1800),
            point(12000, 1400, 1400),
            point(6000, -3000, 700),
            point(11000, 2000, 1800),
            point(13000, -2000, 1300),
        ]

        self.target_pos = point(0, 200, 5)

        # 惩罚机制参数
        self.penalty_factor = penalty_factor  # 惩罚因子
        self.min_blocked_threshold = min_blocked_threshold  # 认为完全没有遮蔽的阈值

        # 奖励机制参数
        self.balance_reward = balance_reward  # 遮蔽时间平衡奖励因子
        self.non_overlap_reward = non_overlap_reward  # 时间段不重叠奖励因子
        self.balance_tolerance = balance_tolerance  # 认为遮蔽时间平衡的容忍度（秒）

        self.setup_nsga2()

    def setup_nsga2(self):
        """设置NSGA-II算法"""
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # 多目标适应度：(M1遮蔽时长, M2遮蔽时长, M3遮蔽时长, 总遮蔽时长)
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 2.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", _evaluate_individual_global)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selNSGA2)

    def create_individual(self):
        """
        创建个体
        编码：每架无人机 [speed, angle, t_first, dt1_2, dt2_3, delay1, delay2, delay3]
        总共 5 * 8 = 40 个参数，所有无人机固定投放3枚烟幕弹
        """
        individual = []

        for drone_idx in range(5):
            # 速度 (70-140 m/s)
            speed = random.uniform(90, 130)

            # 角度 - 全部随机学习
            angle = random.uniform(0, 2 * math.pi)

            # 投放时间参数：首次投放时间 + 两个间隔时间
            t_first = random.uniform(0, 8)  # 首次投放时间
            dt1_2 = random.uniform(1.0, 5.0)  # 第一次到第二次的间隔
            dt2_3 = random.uniform(1.0, 5.0)  # 第二次到第三次的间隔

            # 延迟时间
            delay1 = random.uniform(0.5, 8)
            delay2 = random.uniform(0.5, 8)
            delay3 = random.uniform(0.5, 8)

            individual.extend(
                [speed, angle, t_first, dt1_2, dt2_3, delay1, delay2, delay3]
            )

        return creator.Individual(individual)

    def decode_individual(self, individual):
        """解码个体为仿真参数"""
        drops_data = []

        for drone_idx in range(5):
            base_idx = drone_idx * 8  # 每架无人机8个参数
            speed = individual[base_idx]
            angle = individual[base_idx + 1]
            t_first = individual[base_idx + 2]
            dt1_2 = individual[base_idx + 3]
            dt2_3 = individual[base_idx + 4]
            delay1 = individual[base_idx + 5]
            delay2 = individual[base_idx + 6]
            delay3 = individual[base_idx + 7]

            # 计算实际投放时间
            t1 = t_first
            t2 = t1 + dt1_2
            t3 = t2 + dt2_3

            # 所有无人机固定投放3枚烟幕弹
            drone_drops = [
                {"t": t1, "delay": delay1},
                {"t": t2, "delay": delay2},
                {"t": t3, "delay": delay3},
            ]

            drops_data.append(
                {
                    "drone_idx": drone_idx,
                    "speed": speed,
                    "angle": angle,
                    "drops": drone_drops,
                }
            )

        return drops_data

    def calculate_rewards(self, missile_blocked_times, results):
        """计算奖励函数"""
        total_reward = 0.0
        individual_rewards = [0.0, 0.0, 0.0]  # 为每个导弹单独计算奖励

        # 1. 遮蔽时间平衡奖励
        balance_reward = self.calculate_balance_reward(missile_blocked_times)
        
        # 2. 时间段不重叠奖励 - 为每个导弹单独计算
        missile_non_overlap_rewards = self.calculate_individual_non_overlap_rewards(results)

        # 平衡奖励平分给所有有效遮蔽的导弹
        valid_missile_count = sum(1 for t in missile_blocked_times if t > self.min_blocked_threshold)
        if valid_missile_count > 0:
            balance_reward_per_missile = balance_reward / valid_missile_count
            for i in range(3):
                if missile_blocked_times[i] > self.min_blocked_threshold:
                    individual_rewards[i] += balance_reward_per_missile

        # 不重叠奖励直接分配给对应导弹
        for i in range(3):
            individual_rewards[i] += missile_non_overlap_rewards[i]

        total_reward = sum(individual_rewards)
        return total_reward, balance_reward, sum(missile_non_overlap_rewards), individual_rewards

    def calculate_balance_reward(self, missile_blocked_times):
        """计算遮蔽时间平衡奖励 - 修复版本"""
        # 过滤掉完全没有遮蔽的导弹
        valid_times = [
            t for t in missile_blocked_times if t > self.min_blocked_threshold
        ]

        # 至少需要2个导弹有遮蔽才能谈平衡
        if len(valid_times) < 2:
            return 0.0

        # 计算遮蔽时间的标准差和平均值
        mean_time = np.mean(valid_times)
        std_time = np.std(valid_times)
        
        # 相对标准差（变异系数）
        if mean_time > 0:
            cv = std_time / mean_time
            # 使用更严格的平衡标准
            balance_tolerance_ratio = self.balance_tolerance  # 相对标准差小于设定值才认为平衡
            
            if cv <= balance_tolerance_ratio:
                # 奖励与平衡程度和平均遮蔽时间成正比
                balance_score = max(0, (balance_tolerance_ratio - cv) / balance_tolerance_ratio)
                reward = self.balance_reward * balance_score * mean_time * len(valid_times)
                return reward

        return 0.0

    def calculate_individual_non_overlap_rewards(self, results):
        """为每个导弹单独计算不重叠奖励"""
        missile_rewards = [0.0, 0.0, 0.0]
        
        for i in range(3):
            missile_key = f"missile_{i}"
            if missile_key not in results:
                continue
                
            intervals = results[missile_key].get("blocked_intervals", [])
            
            # 只有当存在多个区间时才计算不重叠奖励
            if len(intervals) < 2:
                continue
            
            # 确保所有区间都有有效的开始和结束时间
            valid_intervals = []
            for interval in intervals:
                if len(interval) >= 2 and interval[1] > interval[0]:
                    valid_intervals.append((interval[0], interval[1]))
            
            if len(valid_intervals) < 2:
                continue
            
            # 按开始时间排序
            valid_intervals.sort(key=lambda x: x[0])
            
            # 计算该导弹的重叠程度
            missile_overlap_time = 0.0
            missile_total_time = 0.0
            
            for interval in valid_intervals:
                missile_total_time += interval[1] - interval[0]
            
            # 计算所有区间对的重叠（不仅仅是相邻的）
            for j in range(len(valid_intervals)):
                for k in range(j + 1, len(valid_intervals)):
                    start1, end1 = valid_intervals[j]
                    start2, end2 = valid_intervals[k]
                    
                    # 计算重叠时间
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    
                    if overlap_start < overlap_end:
                        missile_overlap_time += overlap_end - overlap_start
            
            # 计算该导弹的不重叠奖励
            if missile_total_time > 0:
                overlap_ratio = min(1.0, missile_overlap_time / missile_total_time)
                non_overlap_ratio = max(0, 1 - overlap_ratio)
                # 只有当不重叠程度较高时才给予奖励
                if non_overlap_ratio > 0.7:  # 设置一个阈值
                    missile_rewards[i] = self.non_overlap_reward * non_overlap_ratio * missile_total_time
        
        return missile_rewards

    def calculate_non_overlap_reward(self, results):
        """计算时间段不重叠奖励 - 只对单个导弹内部的时间段重叠进行评估"""
        total_reward = 0.0
        
        # 对每个导弹分别计算其内部时间段的不重叠奖励
        for i in range(3):
            missile_key = f"missile_{i}"
            if missile_key not in results:
                continue
                
            intervals = results[missile_key].get("blocked_intervals", [])
            if len(intervals) < 2:
                # 如果该导弹只有一个或没有遮蔽区间，给予基础奖励
                if len(intervals) == 1 and len(intervals[0]) >= 2:
                    interval_time = intervals[0][1] - intervals[0][0]
                    total_reward += self.non_overlap_reward * interval_time
                continue
            
            # 确保所有区间都有有效的开始和结束时间
            valid_intervals = []
            for interval in intervals:
                if len(interval) >= 2 and interval[1] > interval[0]:
                    valid_intervals.append((interval[0], interval[1]))
            
            if len(valid_intervals) < 2:
                continue
            
            # 按开始时间排序
            valid_intervals.sort(key=lambda x: x[0])
            
            # 计算该导弹的重叠程度
            missile_overlap_time = 0.0
            missile_total_time = 0.0
            
            for interval in valid_intervals:
                missile_total_time += interval[1] - interval[0]
            
            # 计算相邻区间的重叠
            for j in range(len(valid_intervals) - 1):
                start1, end1 = valid_intervals[j]
                start2, end2 = valid_intervals[j + 1]
                
                # 计算重叠时间
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                
                if overlap_start < overlap_end:
                    missile_overlap_time += overlap_end - overlap_start
            
            # 计算该导弹的不重叠奖励
            if missile_total_time > 0:
                overlap_ratio = missile_overlap_time / missile_total_time
                non_overlap_ratio = max(0, 1 - overlap_ratio)
                missile_reward = self.non_overlap_reward * non_overlap_ratio * missile_total_time
                total_reward += missile_reward
        
        return total_reward

    def evaluate_individual(self, individual):
        """评估个体 - 多目标评估"""
        try:
            drops_data = self.decode_individual(individual)

            # 检查约束条件
            if not self.check_constraints(drops_data):
                return (0.0, 0.0, 0.0, 0.0)

            # 构建仿真参数
            missiles = self.missiles.copy()
            smokes = []

            for drone_data in drops_data:
                drone_idx = drone_data["drone_idx"]
                drone_pos = self.drone_positions[drone_idx]
                speed = drone_data["speed"]
                angle = drone_data["angle"]

                vx, vy = speed * math.cos(angle), speed * math.sin(angle)
                v = point(vx, vy, 0)
                fy = drone(drone_pos, v)

                for drop in drone_data["drops"]:
                    smoke_obj = fy.drop_smoke(drop["t"], drop["delay"])
                    smokes.append(smoke_obj)

            # 运行仿真
            results = simulate(missiles, smokes, True, 0.01)

            # 计算各导弹的遮蔽时间
            missile_blocked_times = []
            total_blocked_time = 0

            for i in range(3):
                missile_key = f"missile_{i}"
                if missile_key in results:
                    blocked_time = results[missile_key]["total_blocked_time"]
                    missile_blocked_times.append(blocked_time)
                    total_blocked_time += blocked_time
                else:
                    missile_blocked_times.append(0.0)

            # 计算奖励
            total_reward, balance_reward, non_overlap_reward, individual_rewards = self.calculate_rewards(
                missile_blocked_times, results
            )

            # 添加惩罚机制：对完全没有遮蔽的导弹进行惩罚
            unblocked_missiles = sum(
                1
                for time in missile_blocked_times
                if time <= self.min_blocked_threshold
            )

            # 应用惩罚和奖励
            final_missile_times = []
            for i, time in enumerate(missile_blocked_times):
                final_time = time

                # 应用惩罚
                if time <= self.min_blocked_threshold:
                    final_time = time - self.penalty_factor * unblocked_missiles

                # 应用对应的个体奖励
                final_time += individual_rewards[i]

                final_missile_times.append(final_time)

            # 计算最终总遮蔽时间
            final_total_time = total_blocked_time
            if unblocked_missiles > 0:
                final_total_time -= self.penalty_factor * unblocked_missiles * 2
            final_total_time += total_reward

            return (
                final_missile_times[0],
                final_missile_times[1],
                final_missile_times[2],
                final_total_time,
            )

        except Exception as e:
            return (0.0, 0.0, 0.0, 0.0)

    def check_constraints(self, drops_data):
        """检查约束条件"""
        # 检查时间间隔约束
        for drone_data in drops_data:
            drop_times = [drop["t"] for drop in drone_data["drops"]]
            drop_times.sort()
            for i in range(1, len(drop_times)):
                if drop_times[i] - drop_times[i - 1] < 1.0:
                    return False

        # 所有无人机都投放3枚烟幕弹，总数为15枚，无需额外检查
        return True

    def crossover(self, ind1, ind2):
        """交叉算子 - 所有无人机使用相同策略"""
        for drone_idx in range(5):
            base_idx = drone_idx * 8  # 每架无人机8个参数

            # 使用SBX交叉保持参数连续性
            for i in range(8):
                if random.random() < 0.5:
                    param_idx = base_idx + i
                    if param_idx < len(ind1):
                        # 模拟二进制交叉
                        beta = random.random()
                        if beta <= 0.5:
                            beta = (2 * beta) ** (1.0 / (1 + 1))
                        else:
                            beta = (1.0 / (2 * (1 - beta))) ** (1.0 / (1 + 1))

                        parent1_val = ind1[param_idx]
                        parent2_val = ind2[param_idx]

                        ind1[param_idx] = 0.5 * (
                            (1 + beta) * parent1_val + (1 - beta) * parent2_val
                        )
                        ind2[param_idx] = 0.5 * (
                            (1 - beta) * parent1_val + (1 + beta) * parent2_val
                        )

        return ind1, ind2

    def mutate(self, individual):
        """变异算子 - 统一变异策略"""
        mutation_rate = 0.15

        for drone_idx in range(5):
            base_idx = drone_idx * 8  # 每架无人机8个参数

            # 参数变异
            mutation_params = {
                0: (8, 70, 140),  # speed
                1: (0.5, 0, 2 * math.pi),  # angle
                2: (1.5, 0, 12),  # t_first
                3: (0.8, 1.0, 8.0),  # dt1_2
                4: (0.8, 1.0, 8.0),  # dt2_3
                5: (1.0, 0.5, 12),  # delay1
                6: (1.0, 0.5, 12),  # delay2
                7: (1.0, 0.5, 12),  # delay3
            }

            for i, (sigma, min_val, max_val) in mutation_params.items():
                if random.random() < mutation_rate:
                    param_idx = base_idx + i
                    individual[param_idx] += random.gauss(0, sigma)
                    individual[param_idx] = max(
                        min_val, min(max_val, individual[param_idx])
                    )

                    # 角度特殊处理
                    if i == 1:  # angle
                        individual[param_idx] = individual[param_idx] % (2 * math.pi)

        return (individual,)

    def run_nsga2(self, population_size=200, generations=300, max_workers=16):
        """运行NSGA-II算法"""
        global _problem_instance
        _problem_instance = self  # 设置全局实例供multiprocessing使用

        print(
            f"种群大小: {population_size}, 进化代数: {generations}, 进程数: {max_workers}"
        )

        # 设置并行映射
        pool = Pool(processes=max_workers)
        self.toolbox.register("map", pool.map)

        # 初始化种群
        population = self.toolbox.population(n=population_size)

        # 统计信息
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register(
            "avg", lambda fits: [np.mean([fit[i] for fit in fits]) for i in range(4)]
        )
        stats.register(
            "max", lambda fits: [np.max([fit[i] for fit in fits]) for i in range(4)]
        )

        # 记录进化过程
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "avg", "max"

        try:
            # 评估初始种群
            print("评估初始种群...")
            fitnesses = list(self.toolbox.map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            record = stats.compile(population)
            logbook.record(gen=0, evals=len(population), **record)
            print(
                f"Gen 0: 总遮蔽avg={record['avg'][3]:.2f}, max={record['max'][3]:.2f}"
            )

            # 进化循环
            for gen in range(1, generations + 1):
                # 选择父代 - 使用锦标赛选择
                offspring = tools.selTournament(
                    population, len(population), tournsize=3
                )
                offspring = [self.toolbox.clone(ind) for ind in offspring]

                # 交叉和变异
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.8:  # 交叉概率
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < 0.2:  # 变异概率
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # 并行评估后代
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                if invalid_ind:
                    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit

                # NSGA-II选择
                population = self.toolbox.select(
                    population + offspring, population_size
                )

                # 记录统计信息
                record = stats.compile(population)
                logbook.record(gen=gen, evals=len(invalid_ind), **record)

                if gen % 20 == 0:
                    print(
                        f"Gen {gen}: 总遮蔽avg={record['avg'][3]:.2f}, max={record['max'][3]:.2f}"
                    )
                    print(
                        f"  M1遮蔽max={record['max'][0]:.2f}, M2遮蔽max={record['max'][1]:.2f}, M3遮蔽max={record['max'][2]:.2f}"
                    )

        finally:
            # 确保关闭进程池
            pool.close()
            pool.join()

        # 获取帕累托前沿
        pareto_front = tools.sortNondominated(
            population, len(population), first_front_only=True
        )[0]

        print(f"\n进化完成，帕累托前沿包含 {len(pareto_front)} 个解")

        return population, logbook, pareto_front

    def analyze_results(self, pareto_front):
        """分析帕累托前沿结果"""
        print("\n=== 帕累托前沿分析 ===")

        # 按总遮蔽时间排序
        sorted_solutions = sorted(
            pareto_front, key=lambda x: x.fitness.values[3], reverse=True
        )

        print("前10个解按总遮蔽时间排序:")
        for i, ind in enumerate(sorted_solutions[:10]):
            fitness = ind.fitness.values
            print(
                f"解{i+1}: M1={fitness[0]:.2f}s, M2={fitness[1]:.2f}s, M3={fitness[2]:.2f}s, 总计={fitness[3]:.2f}s"
            )

        # 分析最优解
        if len(sorted_solutions) > 0:
            best_solution = sorted_solutions[0]
            print(f"\n最优解详细分析:")
            self.analyze_solution(best_solution)

        return sorted_solutions

    def analyze_solution(self, individual):
        """分析单个解的详细信息"""
        drops_data = self.decode_individual(individual)
        fitness = individual.fitness.values

        print(
            f"适应度: M1={fitness[0]:.3f}s, M2={fitness[1]:.3f}s, M3={fitness[2]:.3f}s, 总计={fitness[3]:.3f}s"
        )

        # 检查是否有惩罚应用
        unblocked_count = sum(1 for f in fitness[:3] if f < 0)
        if unblocked_count > 0:
            print(
                f"注意：此解应用了惩罚机制，有 {unblocked_count} 个导弹的适应度值为负数（表示完全没有遮蔽）"
            )

        # 计算并显示奖励信息
        try:
            missiles = self.missiles.copy()
            smokes = []

            for drone_data in drops_data:
                drone_idx = drone_data["drone_idx"]
                drone_pos = self.drone_positions[drone_idx]
                speed = drone_data["speed"]
                angle = drone_data["angle"]

                vx, vy = speed * math.cos(angle), speed * math.sin(angle)
                v = point(vx, vy, 0)
                fy = drone(drone_pos, v)

                for drop in drone_data["drops"]:
                    smoke_obj = fy.drop_smoke(drop["t"], drop["delay"])
                    smokes.append(smoke_obj)

            results = simulate(missiles, smokes, True, 0.01)

            # 计算原始遮蔽时间
            original_times = []
            for i in range(3):
                missile_key = f"missile_{i}"
                if missile_key in results:
                    blocked_time = results[missile_key]["total_blocked_time"]
                    original_times.append(blocked_time)
                else:
                    original_times.append(0.0)

            # 计算奖励
            total_reward, balance_reward, non_overlap_reward, individual_rewards = self.calculate_rewards(
                original_times, results
            )

            if total_reward > 0:
                print(f"奖励机制:")
                print(
                    f"  平衡奖励: {balance_reward:.3f}s (相对容忍度: {self.balance_tolerance})"
                )
                print(f"  单导弹不重叠奖励: {non_overlap_reward:.3f}s")
                print(f"  总奖励: {total_reward:.3f}s")
                print(f"  各导弹奖励: M1={individual_rewards[0]:.3f}s, M2={individual_rewards[1]:.3f}s, M3={individual_rewards[2]:.3f}s")

                # 分析平衡性
                valid_times = [
                    t for t in original_times if t > self.min_blocked_threshold
                ]
                if len(valid_times) >= 2:
                    std_time = np.std(valid_times)
                    mean_time = np.mean(valid_times)
                    cv = std_time / mean_time if mean_time > 0 else 0
                    print(f"  遮蔽时间变异系数: {cv:.3f} (平均={mean_time:.3f}s, 标准差={std_time:.3f}s)")
        except Exception as e:
            print(f"奖励分析出错: {e}")

        print("\n无人机详细配置:")

        for drone_data in drops_data:
            drone_idx = drone_data["drone_idx"]

            print(f"FY{drone_idx+1} -> 随机学习:")
            print(f"  速度: {drone_data['speed']:.1f} m/s")
            print(f"  角度: {math.degrees(drone_data['angle']):.1f}°")
            print(f"  投放数量: {len(drone_data['drops'])}")

            for j, drop in enumerate(drone_data["drops"]):
                print(f"    第{j+1}枚: t={drop['t']:.2f}s, delay={drop['delay']:.2f}s")

        # 验证解
        self.verify_solution(individual)

    def verify_solution(self, individual):
        """验证解的正确性"""
        try:
            drops_data = self.decode_individual(individual)

            missiles = self.missiles.copy()
            smokes = []

            for drone_data in drops_data:
                drone_idx = drone_data["drone_idx"]
                drone_pos = self.drone_positions[drone_idx]
                speed = drone_data["speed"]
                angle = drone_data["angle"]

                vx, vy = speed * math.cos(angle), speed * math.sin(angle)
                v = point(vx, vy, 0)
                fy = drone(drone_pos, v)

                for drop in drone_data["drops"]:
                    smoke_obj = fy.drop_smoke(drop["t"], drop["delay"])
                    smokes.append(smoke_obj)

            results = simulate(missiles, smokes, False, 0.01)

            print(f"\n验证仿真结果:")
            total_blocked = 0
            for i in range(3):
                missile_key = f"missile_{i}"
                if missile_key in results:
                    blocked_time = results[missile_key]["total_blocked_time"]
                    total_blocked += blocked_time
                    intervals = results[missile_key].get("blocked_intervals", [])
                    print(f"M{i+1}: {blocked_time:.3f}s, 区间: {intervals}")

            print(f"总遮蔽时间: {total_blocked:.3f}s")
            print(f"使用烟幕弹总数: {len(smokes)}")

        except Exception as e:
            print(f"验证时出错: {e}")

    def plot_pareto_front(self, pareto_front, save_path="nsga2_pareto_front.png"):
        """绘制帕累托前沿"""
        if len(pareto_front) == 0:
            return

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 提取适应度值
        fitness_values = np.array([ind.fitness.values for ind in pareto_front])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 各导弹遮蔽时间散点图
        axes[0, 0].scatter(
            fitness_values[:, 0], fitness_values[:, 1], alpha=0.7, c="blue"
        )
        axes[0, 0].set_xlabel("M1 Blocking Time (s)")
        axes[0, 0].set_ylabel("M2 Blocking Time (s)")
        axes[0, 0].set_title("M1 vs M2 Blocking Time")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(
            fitness_values[:, 0], fitness_values[:, 2], alpha=0.7, c="green"
        )
        axes[0, 1].set_xlabel("M1 Blocking Time (s)")
        axes[0, 1].set_ylabel("M3 Blocking Time (s)")
        axes[0, 1].set_title("M1 vs M3 Blocking Time")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(
            fitness_values[:, 1], fitness_values[:, 2], alpha=0.7, c="red"
        )
        axes[1, 0].set_xlabel("M2 Blocking Time (s)")
        axes[1, 0].set_ylabel("M3 Blocking Time (s)")
        axes[1, 0].set_title("M2 vs M3 Blocking Time")
        axes[1, 0].grid(True, alpha=0.3)

        # 总遮蔽时间分布
        axes[1, 1].hist(fitness_values[:, 3], bins=20, alpha=0.7, color="purple")
        axes[1, 1].set_xlabel("Total Blocking Time (s)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Total Blocking Time Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"帕累托前沿图已保存为 '{save_path}'")


def run_nsga2_problem5(
    penalty_factor=0.8,
    min_blocked_threshold=0.01,
    balance_reward=0.1,  # 降低默认值
    non_overlap_reward=0.05,  # 降低默认值
    balance_tolerance=0.2,
):
    """运行NSGA-II求解问题5"""
    # 创建算法实例
    nsga2 = NSGAII_Problem5(
        penalty_factor=penalty_factor,
        min_blocked_threshold=min_blocked_threshold,
        balance_reward=balance_reward,
        non_overlap_reward=non_overlap_reward,
        balance_tolerance=balance_tolerance,
    )

    print(
        f"惩罚机制参数: penalty_factor={penalty_factor}, min_blocked_threshold={min_blocked_threshold}"
    )
    print(
        f"奖励机制参数: balance_reward={balance_reward}, single_missile_non_overlap_reward={non_overlap_reward}, balance_tolerance={balance_tolerance}"
    )

    # 运行算法
    population, logbook, pareto_front = nsga2.run_nsga2(
        population_size=500, generations=1000, max_workers=16
    )

    # 分析结果
    best_solutions = nsga2.analyze_results(pareto_front)

    # 绘制帕累托前沿
    nsga2.plot_pareto_front(pareto_front)

    return nsga2, population, logbook, pareto_front, best_solutions


if __name__ == "__main__":
    # 设置随机种子以便复现
    random.seed(42)
    np.random.seed(42)

    # 运行NSGA-II算法
    nsga2, population, logbook, pareto_front, best_solutions = run_nsga2_problem5()
