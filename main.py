from object import point, missile, drone, smoke

debug = 0


def simulate(missiles, smokes, time, step):
    results = {}

    for i in range(len(missiles)):
        results[f"missile_{i}"] = {
            "total_blocked_time": 0.0,
            "blocked_intervals": [],
            "blocked_status": [],
        }

    if debug:
        f = open("debug.log", "w")

    current_time = 0.0
    while current_time <= time:
        for i, missile_obj in enumerate(missiles):
            missile_key = f"missile_{i}"

            is_blocked = missile_obj.is_blocked(current_time, smokes)
            results[missile_key]["blocked_status"].append(
                {"time": current_time, "blocked": is_blocked}
            )

        if debug:
            f.write(f"Time: {current_time:.2f}\n")
            for i, missile_obj in enumerate(missiles):
                missile_key = f"missile_{i}"
                missile_pos = missile_obj.getpos(current_time)
                blocked = results[missile_key]["blocked_status"][-1]["blocked"]
                f.write(
                    f"  {missile_key}: Pos({missile_pos.x:.2f}, {missile_pos.y:.2f}, {missile_pos.z:.2f}) Blocked: {blocked}\n"
                )
            for i, smoke_obj in enumerate(smokes):
                smoke_pos = smoke_obj.getpos(current_time)
                if smoke_pos:
                    f.write(
                        f"  Smoke_{i}: Pos({smoke_pos.x:.2f}, {smoke_pos.y:.2f}, {smoke_pos.z:.2f})\n"
                    )
            f.write("\n")

        current_time += step

    if debug:
        f.close()

    for missile_key in results.keys():
        blocked_status = results[missile_key]["blocked_status"]
        intervals = []
        current_interval_start = None

        for status in blocked_status:
            if status["blocked"] and current_interval_start is None:
                current_interval_start = status["time"]
            elif not status["blocked"] and current_interval_start is not None:
                intervals.append((current_interval_start, status["time"]))
                current_interval_start = None

        if current_interval_start is not None:
            intervals.append((current_interval_start, time))

        total_blocked = sum(end - start for start, end in intervals)

        results[missile_key]["blocked_intervals"] = intervals
        results[missile_key]["total_blocked_time"] = total_blocked

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
    results = simulate(missiles, smokes, 30, 0.01)

    print("问题1结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")


def problem2(v, t, delay):
    missiles = [m1]
    fy1 = drone(fy_pos[0], v)
    smokes = []
    smokes.append(fy1.drop_smoke(t, delay))
    results = simulate(missiles, smokes, 50, 0.01)

    print("问题2结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")


def problem3(v, t, delay):
    missiles = [m1]
    fy1 = drone(fy_pos[0], v)
    smokes = []
    for i in range(3):
        smokes.append(fy1.drop_smoke(t[i], delay[i]))
    results = simulate(missiles, smokes, 90, 0.01)

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
    results = simulate(missiles, smokes, 90, 0.01)

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
    results = simulate(missiles, smokes, 90, 0.01)

    print("问题5结果:")
    print(f"M1被遮挡总时间: {results['missile_0']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_0']['blocked_intervals']}")
    print(f"M2被遮挡总时间: {results['missile_1']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_1']['blocked_intervals']}")
    print(f"M3被遮挡总时间: {results['missile_2']['total_blocked_time']:.2f} 秒")
    print(f"遮挡时间段: {results['missile_2']['blocked_intervals']}")


def main():
    problem1()


if __name__ == "__main__":
    main()
