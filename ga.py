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
    """评估函数"""
    try:
        # 个体参数
        speed1, angle1, t1, delay1 = individual[0:4]
        speed2, angle2, t2, delay2 = individual[4:8]
        speed3, angle3, t3, delay3 = individual[8:12]

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

        vx1, vy1 = speed1 * math.cos(angle1), speed1 * math.sin(angle1)
        vx2, vy2 = speed2 * math.cos(angle2), speed2 * math.sin(angle2)
        vx3, vy3 = speed3 * math.cos(angle3), speed3 * math.sin(angle3)

        v = [point(vx1, vy1, 0), point(vx2, vy2, 0), point(vx3, vy3, 0)]

        missiles = [m1]

        fy = []
        for i in range(3):
            fy.append(drone(fy_pos[i], v[i]))

        smokes = []
        smokes.append(fy[0].drop_smoke(t1, delay1))
        smokes.append(fy[1].drop_smoke(t2, delay2))
        smokes.append(fy[2].drop_smoke(t3, delay3))

        results = simulate(missiles, smokes, True, 0.01)
        blocked_time = results["missile_0"]["total_blocked_time"]

        return (blocked_time,)

    except Exception as e:
        return (0.0,)


def setup_ga():
    """设置遗传算法"""
    # 创建适应度类和个体类
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化遮挡时间
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # 无人机: [speed, angle, t, delay]
    def create_individual():
        individual = []
        for i in range(3):
            speed = random.uniform(70, 140)

            drone_pos = fy_pos[i]
            missile_drone_y = m1.y - drone_pos.y
            missile_drone_x = m1.x - drone_pos.x
            target_angle = math.atan2(missile_drone_y, missile_drone_x)

            if random.random() < 0.3:
                angle = target_angle + random.uniform(-math.pi / 2, math.pi / 2)
            else:
                angle = random.uniform(0, 2 * math.pi)

            t = random.uniform(0, 15)
            delay = random.uniform(0, 15)

            individual.extend([speed, angle, t, delay])
        return creator.Individual(individual)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_problem4)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_individual, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=5)

    return toolbox


def mutate_individual(individual, indpb):
    """突变函数，使用不同的突变强度"""
    mutation_strengths = {0: 10, 1: 0.5, 2: 2.0, 3: 3.0}

    for i in range(len(individual)):
        if random.random() < indpb:
            param_type = i % 4
            sigma = mutation_strengths[param_type]

            if param_type == 0:  # speed
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(70, min(140, individual[i]))
            elif param_type == 1:  # angle
                individual[i] += random.gauss(0, sigma)
                individual[i] = (individual[i] + 2 * math.pi) % (2 * math.pi)
            elif param_type == 2:  # t
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(0, min(20, individual[i]))
            elif param_type == 3:  # delay
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(0, min(10, individual[i]))

    return (individual,)


def run_ga_problem4():
    """运行改进的遗传算法求解问题4"""
    toolbox = setup_ga()

    # 并行评估
    pool = Pool(processes=16)
    toolbox.register("map", pool.map)

    population_size = 300
    generations = 300
    cx_prob = 0.8
    mut_prob = 0.4

    print(f"种群大小: {population_size}, 进化代数: {generations}")
    print(f"交叉概率: {cx_prob}, 变异概率: {mut_prob}")

    pop = toolbox.population(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    hall_of_fame = tools.HallOfFame(5)

    # 自适应进化策略
    def custom_eaSimple(
        population, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose
    ):
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        # 评估初始种群
        fitnesses = list(toolbox.map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        # 进化循环
        for gen in range(1, ngen + 1):
            if gen > 50 and gen % 10 == 0:
                # 检查停滞情况
                recent_max = [logbook[i]["max"] for i in range(max(0, gen - 10), gen)]
                if len(set(f"{x:.3f}" for x in recent_max)) <= 2:
                    current_mutpb = min(0.7, mutpb * 1.5)
                    print(f"第{gen}代检测到停滞，增加变异概率到 {current_mutpb:.2f}")
                else:
                    current_mutpb = mutpb
            else:
                current_mutpb = mutpb

            # 选择下一代
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # 交叉和变异
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < current_mutpb:
                    # 对于停滞情况，增加大变异概率
                    if gen > 50 and random.random() < 0.1:
                        for i in range(0, len(mutant), 4):
                            if (
                                random.random() < 0.3
                            ):  # 30%概率重新初始化一架无人机的参数
                                mutant[i] = random.uniform(70, 140)
                                mutant[i + 1] = random.uniform(0, 2 * math.pi)
                                mutant[i + 2] = random.uniform(0, 20)
                                mutant[i + 3] = random.uniform(0, 20)
                    else:
                        toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 评估个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

            if halloffame is not None:
                halloffame.update(population)

            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    try:
        pop, logbook = custom_eaSimple(
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

    print(f"\n最终结果:")
    print(f"最大遮挡时间: {best_fitness:.3f} 秒")

    # 显示前5个最优解
    print(f"\nHall of Fame (前5个最优解):")
    for i, ind in enumerate(hall_of_fame):
        print(f"第{i+1}名: {ind.fitness.values[0]:.3f}秒")

    # 最优参数
    speed1, angle1, t1, delay1 = best_individual[0:4]
    speed2, angle2, t2, delay2 = best_individual[4:8]
    speed3, angle3, t3, delay3 = best_individual[8:12]

    print(f"\n最优解参数:")
    print(f"无人机FY1 (位置: {fy_pos[0].x}, {fy_pos[0].y}, {fy_pos[0].z}):")
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

    results = simulate(missiles, smokes, False, 0.01)

    print(f"\n验证结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.3f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")

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
