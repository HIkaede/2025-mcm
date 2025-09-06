import math
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
from object import point, missile, drone, smoke
import libsimulate
from main import m1, fy_pos, simulate


def _plot_convergence_results(result, best_blocked_time, filename, title):
    """绘制收敛结果图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：收敛过程
    plot_convergence(result, ax=ax1)
    ax1.set_title(f"{title} - Bayesian Optimization Convergence")
    ax1.set_xlabel("Number of Function Evaluations")
    ax1.set_ylabel("Objective Function Value (Negative Blocking Time)")
    ax1.grid(True, alpha=0.3)

    # 右图：参数变化
    func_vals = result.func_vals
    evaluations = range(1, len(func_vals) + 1)

    ax2.plot(evaluations, [-val for val in func_vals], "b-", alpha=0.7, linewidth=1)
    ax2.scatter(evaluations, [-val for val in func_vals], c="red", s=20, alpha=0.6)
    ax2.set_title(f"{title} - Blocking Time Progress")
    ax2.set_xlabel("Evaluation Number")
    ax2.set_ylabel("Blocking Time (seconds)")
    ax2.grid(True, alpha=0.3)

    # 标记最佳点
    best_idx = list(func_vals).index(result.fun)
    ax2.scatter(
        [best_idx + 1],
        [best_blocked_time],
        c="green",
        s=100,
        marker="*",
        label=f"Best: {best_blocked_time:.2f}s",
        zorder=5,
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"收敛图已保存为 '{filename}'")


def objective_function_problem2(params):
    """目标函数：返回负的被遮挡时间"""
    speed, angle, t, delay = params

    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    v = point(vx, vy, 0)

    missiles = [m1]
    fy1 = drone(fy_pos[0], v)
    smokes = [fy1.drop_smoke(t, delay)]

    results = simulate(missiles, smokes, False, 0.001)
    blocked_time = results["missile_0"]["total_blocked_time"]
    return -blocked_time


def bayesian_optimize_problem2():
    """使用贝叶斯优化求解问题2"""
    dimensions = [
        Real(70.0, 140.0, name="speed"),
        Real(0.0, 2 * math.pi, name="angle"),
        Real(0.0, 8.0, name="t"),
        Real(0.0, 6.0, name="delay"),
    ]

    result = gp_minimize(
        func=objective_function_problem2,
        dimensions=dimensions,
        n_calls=100,
        n_initial_points=20,
        random_state=42,
        verbose=True,
    )

    best_speed, best_angle, best_t, best_delay = result.x
    best_vx = best_speed * math.cos(best_angle)
    best_vy = best_speed * math.sin(best_angle)
    best_blocked_time = -result.fun

    print(
        f"最优速度: speed={best_speed:.2f}, angle={best_angle:.2f}弧度 ({math.degrees(best_angle):.1f}度)"
    )
    print(f"速度分量: vx={best_vx:.2f}, vy={best_vy:.2f}")
    print(f"最优投放时间: {best_t:.2f}s")
    print(f"最优延迟时间: {best_delay:.2f}s")
    print(f"最大遮挡时间: {best_blocked_time:.2f}s")

    # 验证结果
    v_optimal = point(best_vx, best_vy, 0)
    missiles = [m1]
    fy1 = drone(fy_pos[0], v_optimal)
    smokes = [fy1.drop_smoke(best_t, best_delay)]
    results = simulate(missiles, smokes, False, 0.01)

    print(
        f"验证结果 - M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f}秒"
    )
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")

    # _plot_convergence_results(result, best_blocked_time, "problem2.png", "Problem 2")
    return result


def objective_function_problem3(params):
    """目标函数：返回负的被遮挡时间"""
    speed, angle, t1, dt2, dt3, delay1, delay2, delay3 = params

    t2 = t1 + dt2
    t3 = t2 + dt3

    # 时间约束检查
    if t3 + max(delay1, delay2, delay3) > 25:
        return 0

    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    v = point(vx, vy, 0)

    missiles = [m1]
    fy1 = drone(fy_pos[0], v)
    smokes = [
        fy1.drop_smoke(t1, delay1),
        fy1.drop_smoke(t2, delay2),
        fy1.drop_smoke(t3, delay3),
    ]

    results = simulate(missiles, smokes, True, 0.001)
    blocked_time = results["missile_0"]["total_blocked_time"]
    return -blocked_time


def bayesian_optimize_problem3():
    """使用贝叶斯优化求解问题3"""
    dimensions = [
        Real(125, 140.0, name="speed"),
        Real(0.0, 2, name="angle"),
        Real(0.0, 1.0, name="t1"),
        Real(1.0, 2.0, name="dt2"),
        Real(1.0, 2.0, name="dt3"),
        Real(0.0, 2.0, name="delay1"),
        Real(0.0, 2.0, name="delay2"),
        Real(0.0, 2.0, name="delay3"),
    ]

    result = gp_minimize(
        func=objective_function_problem3,
        dimensions=dimensions,
        n_calls=150,
        n_initial_points=40,
        random_state=42,
        verbose=True,
    )

    (
        best_speed,
        best_angle,
        best_t1,
        best_dt2,
        best_dt3,
        best_delay1,
        best_delay2,
        best_delay3,
    ) = result.x

    best_vx = best_speed * math.cos(best_angle)
    best_vy = best_speed * math.sin(best_angle)
    best_blocked_time = -result.fun
    best_t2 = best_t1 + best_dt2
    best_t3 = best_t2 + best_dt3

    print(
        f"最优速度: speed={best_speed:.2f}, angle={best_angle:.2f}弧度 ({math.degrees(best_angle):.1f}度)"
    )
    print(f"速度分量: vx={best_vx:.2f}, vy={best_vy:.2f}")
    print(f"投放策略:")
    print(f"  第一次投放: t1={best_t1:.2f}s, 延迟={best_delay1:.2f}s")
    print(
        f"  第二次投放: t2={best_t2:.2f}s (间隔{best_dt2:.2f}s), 延迟={best_delay2:.2f}s"
    )
    print(
        f"  第三次投放: t3={best_t3:.2f}s (间隔{best_dt3:.2f}s), 延迟={best_delay3:.2f}s"
    )
    print(f"最大遮挡时间: {best_blocked_time:.2f}秒")

    # 验证结果
    v_optimal = point(best_vx, best_vy, 0)
    missiles = [m1]
    fy1 = drone(fy_pos[0], v_optimal)
    smokes = [
        fy1.drop_smoke(best_t1, best_delay1),
        fy1.drop_smoke(best_t2, best_delay2),
        fy1.drop_smoke(best_t3, best_delay3),
    ]
    results = simulate(missiles, smokes, True, 0.01)

    print(
        f"\n验证结果 - M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f}秒"
    )
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")

    _plot_convergence_results(result, best_blocked_time, "problem3.png", "Problem 3")
    return result


if __name__ == "__main__":
    result = bayesian_optimize_problem2()
    # result = bayesian_optimize_problem3()
