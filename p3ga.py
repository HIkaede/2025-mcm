import random
import math
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
from object import point, missile, drone, smoke
import libsimulate
from main import m1, m2, m3, fy_pos, simulate


def evaluate_problem3(individual):
    """评估函数 - 问题3：FY1投放3枚烟幕干扰弹对M1的干扰"""
    try:
        # 个体参数: [speed, angle, t1, dt2, dt3, delay1, delay2, delay3]
        speed, angle, t1, dt2, dt3, delay1, delay2, delay3 = individual

        if not (70 <= speed <= 140):
            return (0.0,)

        t2 = t1 + dt2
        t3 = t2 + dt3
        
        if not (0 <= t1 <= 15 and 1 <= dt2 <= 10 and 1 <= dt3 <= 10):
            return (0.0,)
        
        if not (0 <= delay1 <= 20 and 0 <= delay2 <= 20 and 0 <= delay3 <= 20):
            return (0.0,)

        max_time = max(t1 + delay1, t2 + delay2, t3 + delay3)
        if max_time > 40:
            return (0.0,)

        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        v = point(vx, vy, 0)

        missiles = [m1]
        fy1 = drone(fy_pos[0], v)

        smokes = [
            fy1.drop_smoke(t1, delay1),
            fy1.drop_smoke(t2, delay2),
            fy1.drop_smoke(t3, delay3)
        ]

        results = simulate(missiles, smokes, True, 0.01)
        blocked_time = results["missile_0"]["total_blocked_time"]

        return (blocked_time,)

    except Exception as e:
        return (0.0,)


def setup_ga_problem3():
    """设置遗传算法"""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化遮挡时间
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def create_individual():
        """创建个体：[speed, angle, t1, dt2, dt3, delay1, delay2, delay3]"""
        speed = random.uniform(70, 140)
        
        # 计算朝向目标的角度作为参考
        drone_pos = fy_pos[0]  # FY1位置
        missile_drone_y = m1.y - drone_pos.y
        missile_drone_x = m1.x - drone_pos.x
        target_angle = math.atan2(missile_drone_y, missile_drone_x)
        
        if random.random() < 0.4:  # 40%概率朝向目标方向
            angle = target_angle + random.uniform(-math.pi/3, math.pi/3)
        else:
            angle = random.uniform(0, 2 * math.pi)
        
        # 投放时间参数
        t1 = random.uniform(0, 10)
        dt2 = random.uniform(1, 10)  # 第二次投放间隔
        dt3 = random.uniform(1, 10)  # 第三次投放间隔
        
        # 延迟时间
        delay1 = random.uniform(0, 10)
        delay2 = random.uniform(0, 10)
        delay3 = random.uniform(0, 10)

        return creator.Individual([speed, angle, t1, dt2, dt3, delay1, delay2, delay3])

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_problem3)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_individual_problem3, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=5)

    return toolbox


def mutate_individual_problem3(individual, indpb):
    """突变函数"""
    # 参数类型对应的突变强度
    mutation_strengths = {
        0: 10,    # speed
        1: 0.5,   # angle
        2: 1.0,   # t1
        3: 1.0,   # dt2
        4: 1.0,   # dt3
        5: 2.0,   # delay1
        6: 2.0,   # delay2
        7: 2.0    # delay3
    }

    for i in range(len(individual)):
        if random.random() < indpb:
            sigma = mutation_strengths[i]

            if i == 0:  # speed
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(70, min(140, individual[i]))
            elif i == 1:  # angle
                individual[i] += random.gauss(0, sigma)
                individual[i] = (individual[i] + 2 * math.pi) % (2 * math.pi)
            elif i == 2:  # t1
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(0, min(30, individual[i]))
            elif i in [3, 4]:  # dt2, dt3
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(1, min(30, individual[i]))
            else:  # delay1, delay2, delay3
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(0, min(20, individual[i]))

    return (individual,)


def run_ga_problem3():
    """遗传算法求解问题3"""
    toolbox = setup_ga_problem3()

    # 并行评估
    pool = Pool(processes=16)
    toolbox.register("map", pool.map)

    population_size = 100
    generations = 10000
    cx_prob = 0.8
    mut_prob = 0.5

    print(f"问题3 - 遗传算法参数:")
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
    def custom_eaSimple(population, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose):
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
            # 自适应变异概率
            if gen > 30 and gen % 10 == 0:
                # 检查停滞情况
                recent_max = [logbook[i]["max"] for i in range(max(0, gen - 10), gen)]
                if len(set(f"{x:.3f}" for x in recent_max)) <= 2:
                    current_mutpb = min(0.6, mutpb * 1.5)
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
                        if random.random() < 0.3:
                            mutant[0] = random.uniform(70, 140)  # speed
                            mutant[1] = random.uniform(0, 2 * math.pi)  # angle
                            mutant[2] = random.uniform(0, 8)  # t1
                            mutant[3] = random.uniform(1, 6)  # dt2
                            mutant[4] = random.uniform(1, 6)  # dt3
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

    print(f"\n问题3最终结果:")
    print(f"最大遮挡时间: {best_fitness:.3f} 秒")

    print(f"\nHall of Fame (前5个最优解):")
    for i, ind in enumerate(hall_of_fame):
        print(f"第{i+1}名: {ind.fitness.values[0]:.3f}秒")

    speed, angle, t1, dt2, dt3, delay1, delay2, delay3 = best_individual
    t2 = t1 + dt2
    t3 = t2 + dt3

    print(f"\n最优解参数:")
    print(f"无人机FY1 (位置: {fy_pos[0].x}, {fy_pos[0].y}, {fy_pos[0].z}):")
    print(f"  速度: {speed:.2f} m/s, 角度: {math.degrees(angle):.1f}°")
    print(f"  速度分量: vx={speed*math.cos(angle):.2f}, vy={speed*math.sin(angle):.2f}")
    
    print(f"\n投放策略:")
    print(f"  第1枚烟幕弹: t1={t1:.2f}s, 延迟={delay1:.2f}s, 起爆时间={t1+delay1:.2f}s")
    print(f"  第2枚烟幕弹: t2={t2:.2f}s (间隔{dt2:.2f}s), 延迟={delay2:.2f}s, 起爆时间={t2+delay2:.2f}s")
    print(f"  第3枚烟幕弹: t3={t3:.2f}s (间隔{dt3:.2f}s), 延迟={delay3:.2f}s, 起爆时间={t3+delay3:.2f}s")

    # 验证最优解
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    v = point(vx, vy, 0)

    missiles = [m1]
    fy1 = drone(fy_pos[0], v)
    smokes = [
        fy1.drop_smoke(t1, delay1),
        fy1.drop_smoke(t2, delay2),
        fy1.drop_smoke(t3, delay3)
    ]

    results = simulate(missiles, smokes, False, 0.01)

    print(f"\n验证结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.3f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")

    plot_evolution_problem3(logbook)

    return best_individual, logbook


def plot_evolution_problem3(logbook):
    """绘制遗传算法进化过程"""
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    fit_avg = logbook.select("avg")
    fit_min = logbook.select("min")

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(gen, fit_max, "r-", linewidth=2, label="Maximum")
    ax.plot(gen, fit_avg, "b-", linewidth=2, label="Average")
    ax.plot(gen, fit_min, "g-", linewidth=2, label="Minimum")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Blocking Time (seconds)")
    ax.set_title("Problem 3 - GA Evolution Progress (FY1 with 3 Smoke Bombs)")
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
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig("problem3_ga.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("问题3进化过程图已保存为 'problem3_ga.png'")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    best_solution, logbook = run_ga_problem3()
