from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
from object import point, missile, drone, smoke
import libsimulate
from main import m1, fy_pos, simulate
import math
import matplotlib.pyplot as plt


def objective_function(params):
    """目标函数：返回负的被遮挡时间（因为我们要最大化遮挡时间）"""
    speed, angle, t, delay = params

    # 将速度和角度转换为速度分量
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)

    # 构建速度向量，确保垂直速度为0
    v = point(vx, vy, 0)

    missiles = [m1]
    fy1 = drone(fy_pos[0], v)
    smokes = []
    smokes.append(fy1.drop_smoke(t, delay))
    results = simulate(missiles, smokes, 30, 0.01)

    # 返回负的被遮挡时间（因为gp_minimize是最小化）
    blocked_time = results["missile_0"]["total_blocked_time"]
    return -blocked_time


def bayesian_optimize_problem2():
    """使用贝叶斯优化求解问题2"""
    # 定义搜索空间
    # speed: 速度大小，在70-140之间
    # angle: 速度方向角度（弧度）
    # t: 投放时间
    # delay: 延迟时间
    dimensions = [
        Real(70.0, 140.0, name="speed"),  # 速度大小
        Real(0.0, 2 * math.pi, name="angle"),  # 方向角度
        Real(0.0, 8.0, name="t"),  # 投放时间
        Real(0.0, 6.0, name="delay"),  # 延迟时间
    ]

    # 执行贝叶斯优化
    result = gp_minimize(
        func=objective_function,
        dimensions=dimensions,
        n_calls=80,
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
    print(f"最优投放时间: {best_t:.2f}秒")
    print(f"最优延迟时间: {best_delay:.2f}秒")
    print(f"最大遮挡时间: {best_blocked_time:.2f}秒")

    v_optimal = point(best_vx, best_vy, 0)
    missiles = [m1]
    fy1 = drone(fy_pos[0], v_optimal)
    smokes = []
    smokes.append(fy1.drop_smoke(best_t, best_delay))
    results = simulate(missiles, smokes, 30, 0.01)

    print(
        f"验证结果 - M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f}秒"
    )
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：收敛过程
    plot_convergence(result, ax=ax1)
    ax1.set_title("Bayesian Optimization Convergence")
    ax1.set_xlabel("Number of Function Evaluations")
    ax1.set_ylabel("Objective Function Value (Negative Blocking Time)")
    ax1.grid(True, alpha=0.3)

    # 右图：参数变化
    func_vals = result.func_vals
    evaluations = range(1, len(func_vals) + 1)

    ax2.plot(evaluations, [-val for val in func_vals], "b-", alpha=0.7, linewidth=1)
    ax2.scatter(evaluations, [-val for val in func_vals], c="red", s=20, alpha=0.6)
    ax2.set_title("Blocking Time Progress")
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
    plt.savefig("bayesian_optimization_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("收敛图已保存为 'bayesian_optimization_convergence.png'")


if __name__ == "__main__":
    result = bayesian_optimize_problem2()
