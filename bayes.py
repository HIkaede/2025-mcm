from skopt import gp_minimize
from skopt.space import Real
from object import point, missile, drone, smoke
import libsimulate
from main import m1, fy_pos, simulate
import math


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
        n_calls=100,
        n_initial_points=20,
        random_state=42,
        verbose=True
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


if __name__ == "__main__":
    result = bayesian_optimize_problem2()