from object import point, missile, drone, smoke
import libsimulate


def simulate(missiles, smokes, check_mid, step):
    """使用C++加速的模拟函数"""
    cpp_missiles = []
    for missile_obj in missiles:
        cpp_pos = libsimulate.Point(missile_obj.x, missile_obj.y, missile_obj.z)
        cpp_missiles.append(libsimulate.Missile(cpp_pos))

    cpp_smokes = []
    for smoke_obj in smokes:
        cpp_pos = libsimulate.Point(smoke_obj.x, smoke_obj.y, smoke_obj.z)
        cpp_smokes.append(libsimulate.Smoke(cpp_pos, smoke_obj.start))

    # 调用cpp函数
    cpp_results = libsimulate.simulate_cpp(cpp_missiles, cpp_smokes, check_mid, step)

    results = {}
    for i, cpp_result in enumerate(cpp_results):
        results[f"missile_{i}"] = {
            "total_blocked_time": cpp_result.total_blocked_time,
            "blocked_intervals": cpp_result.blocked_intervals,
        }

    return results


m1 = missile(point(20000, 0, 2000))
m2 = missile(point(19600, 600, 2100))
m3 = missile(point(18000, -600, 1900))


fy_pos = [
    point(17800, 0, 1800),
    point(12000, 1400, 1400),
    point(6000, -3000, 700),
    point(11000, 2000, 1800),
    point(13000, -2000, 1300),
]


def problem1():
    missiles = [m1]
    fy1 = drone(fy_pos[0], point(-120, 0, 0))
    smokes = []
    smokes.append(fy1.drop_smoke(1.5, 3.6))
    results = simulate(missiles, smokes, False, 0.001)

    print("问题1结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.3f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")


def problem2(v, t, delay):
    missiles = [m1]
    fy1 = drone(fy_pos[0], v)
    smokes = []
    smokes.append(fy1.drop_smoke(t, delay))
    results = simulate(missiles, smokes, False, 0.01)

    print("问题2结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")


def problem3(v, t, delay):
    missiles = [m1]
    fy1 = drone(fy_pos[0], v)
    smokes = []
    for i in range(3):
        smokes.append(fy1.drop_smoke(t[i], delay[i]))
    results = simulate(missiles, smokes, True, 0.01)

    print("问题3结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")


def problem4(v, t, delay):
    missiles = [m1]
    fy = []
    for i in range(3):
        fy.append(drone(fy_pos[i], v[i]))
    smokes = []
    for i in range(3):
        smokes.append(fy[i].drop_smoke(t[i], delay[i]))
    results = simulate(missiles, smokes, True, 0.01)

    print("问题4结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")


def problem5(v, t, delay):
    missiles = [m1, m2, m3]
    fy = []
    for i in range(5):
        fy.append(drone(fy_pos[i], v[i]))
    smokes = []
    for i in range(3):
        smokes.append(fy[i].drop_smoke(t[i], delay[i]))
    results = simulate(missiles, smokes, True, 0.001)

    print("问题5结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")
    print(f"M2被遮挡总时间: {results['missile_1']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_1']['blocked_intervals']}")
    print(f"M3被遮挡总时间: {results['missile_2']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_2']['blocked_intervals']}")


if __name__ == "__main__":
    problem1()
