import random
import math
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
from object import point, missile, drone, smoke
import libsimulate
from main import m1, fy_pos, simulate


def evaluate_problem4(individual):
    """
    评估函数：计算3架无人机投放烟幕弹的遮挡效果
    individual: [speed1, angle1, t1, delay1, speed2, angle2, t2, delay2, speed3, angle3, t3, delay3]
    """
    try:
        # 解析个体参数
        speed1, angle1, t1, delay1 = individual[0:4]
        speed2, angle2, t2, delay2 = individual[4:8]
        speed3, angle3, t3, delay3 = individual[8:12]

        # 检查约束条件
        if not (70 <= speed1 <= 140 and 70 <= speed2 <= 140 and 70 <= speed3 <= 140):
            return (0.0,)

        # 时间约束
        max_time = max(t1 + delay1, t2 + delay2, t3 + delay3)
        if max_time > 80:
            return (0.0,)

        if not (0 <= t1 <= 60 and 0 <= t2 <= 60 and 0 <= t3 <= 60):
            return (0.0,)

        if not (0 <= delay1 <= 20 and 0 <= delay2 <= 20 and 0 <= delay3 <= 20):
            return (0.0,)

        # 计算速度向量
        vx1, vy1 = speed1 * math.cos(angle1), speed1 * math.sin(angle1)
        vx2, vy2 = speed2 * math.cos(angle2), speed2 * math.sin(angle2)
        vx3, vy3 = speed3 * math.cos(angle3), speed3 * math.sin(angle3)

        v = [point(vx1, vy1, 0), point(vx2, vy2, 0), point(vx3, vy3, 0)]

        missiles = [m1]

        # 创建3架无人机
        fy = []
        for i in range(3):
            fy.append(drone(fy_pos[i], v[i]))

        # 投放烟幕弹
        smokes = []
        smokes.append(fy[0].drop_smoke(t1, delay1))
        smokes.append(fy[1].drop_smoke(t2, delay2))
        smokes.append(fy[2].drop_smoke(t3, delay3))

        # 运行模拟
        results = simulate(missiles, smokes, True, 0.01)
        blocked_time = results["missile_0"]["total_blocked_time"]

        return (blocked_time,)

    except Exception as e:
        # 如果出现任何错误，返回最差适应度
        return (0.0,)


def setup_ga():
    """设置遗传算法"""
    # 创建适应度类和个体类
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化遮挡时间
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # 参数范围定义
    # 每架无人机: [speed, angle, t, delay]
    # speed: 70-140, angle: 0-2π, t: 0-8, delay: 0-6
    def create_individual():
        individual = []
        for i in range(3):  # 3架无人机
            speed = random.uniform(120, 140)

            # 优化角度选择
            drone_pos = fy_pos[i]
            missile_drone_y = m1.y - drone_pos.y
            missile_drone_x = m1.x - drone_pos.x
            target_angle = math.atan2(missile_drone_y, missile_drone_x)

            if random.random() < 0.6:
                angle = target_angle + random.uniform(-math.pi / 3, math.pi / 3)
            else:
                angle = random.uniform(0, 2 * math.pi)

            # 优化时间参数选择
            t = random.uniform(0, 5)  # 投放时间集中在有效范围
            delay = random.uniform(0, 5)  # 延迟时间在合理范围

            individual.extend([speed, angle, t, delay])
        return creator.Individual(individual)

    # 注册遗传算法操作
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_problem4)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def mutate_individual(individual, indpb):
    """自定义变异函数，考虑参数约束"""
    for i in range(len(individual)):
        if random.random() < indpb:
            param_type = i % 4  # 0:speed, 1:angle, 2:t, 3:delay

            if param_type == 0:  # speed
                individual[i] += random.gauss(0, 8)
                individual[i] = max(70, min(140, individual[i]))
            elif param_type == 1:  # angle
                individual[i] += random.gauss(0, 0.3)
                individual[i] = individual[i] % (2 * math.pi)
            elif param_type == 2:  # t
                individual[i] += random.gauss(0, 0.8)
                individual[i] = max(0, min(10, individual[i]))  # 增加时间上限
            elif param_type == 3:  # delay
                individual[i] += random.gauss(0, 0.6)
                individual[i] = max(0, min(8, individual[i]))  # 增加延迟上限

    return (individual,)


def run_ga_problem4():
    """运行遗传算法求解问题4"""
    toolbox = setup_ga()
    toolbox.register("mutate", mutate_individual, indpb=0.2)

    # 并行评估
    pool = Pool(processes=8)
    toolbox.register("map", pool.map)

    # 遗传算法参数
    population_size = 100
    generations = 150
    cx_prob = 0.7
    mut_prob = 0.3

    print(f"种群大小: {population_size}, 进化代数: {generations}")

    # 创建初始种群
    pop = toolbox.population(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)

    hall_of_fame = tools.HallOfFame(1)

    try:
        pop, logbook = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=cx_prob,
            mutpb=mut_prob,
            ngen=generations,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=True,
        )
    finally:
        # 确保关闭进程池
        pool.close()
        pool.join()

    best_individual = hall_of_fame[0]
    best_fitness = best_individual.fitness.values[0]

    print(f"\n最优解找到!")
    print(f"最大遮挡时间: {best_fitness:.3f} 秒")

    # 最优参数
    speed1, angle1, t1, delay1 = best_individual[0:4]
    speed2, angle2, t2, delay2 = best_individual[4:8]
    speed3, angle3, t3, delay3 = best_individual[8:12]

    print(f"\n无人机FY1 (位置: {fy_pos[0].x}, {fy_pos[0].y}, {fy_pos[0].z}):")
    print(f"  速度: {speed1:.2f} m/s, 角度: {math.degrees(angle1):.1f}°")
    print(
        f"  速度分量: vx={speed1*math.cos(angle1):.2f}, vy={speed1*math.sin(angle1):.2f}"
    )
    print(f"  投放时间: {t1:.2f}s, 延迟: {delay1:.2f}s")

    print(f"\n无人机FY2 (位置: {fy_pos[1].x}, {fy_pos[1].y}, {fy_pos[1].z}):")
    print(f"  速度: {speed2:.2f} m/s, 角度: {math.degrees(angle2):.1f}°")
    print(
        f"  速度分量: vx={speed2*math.cos(angle2):.2f}, vy={speed2*math.sin(angle2):.2f}"
    )
    print(f"  投放时间: {t2:.2f}s, 延迟: {delay2:.2f}s")

    print(f"\n无人机FY3 (位置: {fy_pos[2].x}, {fy_pos[2].y}, {fy_pos[2].z}):")
    print(f"  速度: {speed3:.2f} m/s, 角度: {math.degrees(angle3):.1f}°")
    print(
        f"  速度分量: vx={speed3*math.cos(angle3):.2f}, vy={speed3*math.sin(angle3):.2f}"
    )
    print(f"  投放时间: {t3:.2f}s, 延迟: {delay3:.2f}s")

    # 验证最优解
    v = [
        point(speed1 * math.cos(angle1), speed1 * math.sin(angle1), 0),
        point(speed2 * math.cos(angle2), speed2 * math.sin(angle2), 0),
        point(speed3 * math.cos(angle3), speed3 * math.sin(angle3), 0),
    ]
    t = [t1, t2, t3]
    delay = [delay1, delay2, delay3]

    missiles = [m1]
    fy = []
    for i in range(3):
        fy.append(drone(fy_pos[i], v[i]))

    smokes = []
    for i in range(3):
        smokes.append(fy[i].drop_smoke(t[i], delay[i]))

    results = simulate(missiles, smokes, True, 0.01)

    print(f"\n验证结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.3f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")

    # 绘制进化过程
    plot_evolution(logbook)

    return best_individual, logbook


def plot_evolution(logbook):
    """绘制遗传算法进化过程"""
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    fit_avg = logbook.select("avg")
    fit_min = logbook.select("min")

    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制适应度进化曲线
    ax.plot(gen, fit_max, "r-", linewidth=2, label="Maximum")
    ax.plot(gen, fit_avg, "b-", linewidth=2, label="Average")
    ax.plot(gen, fit_min, "g-", linewidth=2, label="Minimum")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Blocking Time (seconds)")
    ax.set_title("Problem 4 - GA Evolution Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加最终结果文本
    final_max = fit_max[-1]
    ax.text(
        0.02,
        0.98,
        f"Final Best: {final_max:.3f}s",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig("problem4.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("进化过程图已保存为 'problem4.png'")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    best_solution, logbook = run_ga_problem4()
