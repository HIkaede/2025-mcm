#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdint>

struct Point
{
    double x, y, z;

    Point(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    Point operator+(const Point &other) const
    {
        return Point(x + other.x, y + other.y, z + other.z);
    }

    Point operator-(const Point &other) const
    {
        return Point(x - other.x, y - other.y, z - other.z);
    }

    Point operator*(double scalar) const
    {
        return Point(x * scalar, y * scalar, z * scalar);
    }

    double dot(const Point &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    Point cross(const Point &other) const
    {
        return Point(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x);
    }

    double length() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    Point normalize() const
    {
        double l = length();
        if (std::abs(l) < 1e-10)
            return Point(0, 0, 0);
        return Point(x / l, y / l, z / l);
    }

    double dist(const Point &other) const
    {
        return std::sqrt(
            (x - other.x) * (x - other.x) +
            (y - other.y) * (y - other.y) +
            (z - other.z) * (z - other.z));
    }

    double dist2line(const Point &a, const Point &b) const
    {
        Point ab = b - a;
        Point ap = *this - a;
        double ab2 = ab.dot(ab);
        double t = ap.dot(ab) / ab2;
        Point proj = a + ab * t;
        return this->dist(proj);
    }
};

class Missile
{
public:
    Point pos;
    Point v;

    Missile(const Point &position) : pos(position)
    {
        v = pos.normalize() * (-300);
    }

    Point getpos(double t) const
    {
        return Point(
            pos.x + v.x * t,
            pos.y + v.y * t,
            pos.z + v.z * t);
    }

    bool is_blocked(double t, const std::vector<class Smoke> &smoke_list) const;
};

class Smoke
{
public:
    Point pos;
    double start_time;

    Smoke(const Point &position, double start) : pos(position), start_time(start) {}

    bool is_active(double t) const
    {
        return (t - start_time <= 20) && (t >= start_time);
    }

    Point getpos(double t) const
    {
        return Point(pos.x, pos.y, pos.z - 3 * (t - start_time));
    }
};

bool Missile::is_blocked(double t, const std::vector<Smoke> &smoke_list) const
{
    Point missile_pos = getpos(t);
    Point target_center(0, 200, 5);
    constexpr double target_radius = 7.0;

    Point bottom_center(0, 200, 0);
    Point top_center(0, 200, 10);

    constexpr int32_t sample_points = 360;
    constexpr double angle_step = 2.0 * M_PI / sample_points;
    constexpr double smoke_radius = 10.0;

    int32_t active_smoke = 0;
    int32_t blocked_smoke = 0;

    for (const auto &smoke : smoke_list)
    {
        if (!smoke.is_active(t))
            continue;

        active_smoke++;

        const Point smoke_pos = smoke.getpos(t);

        // 检查烟幕是否在导弹和目标之间
        const double missile_to_target_dist = missile_pos.dist(target_center);
        const double smoke_to_target_dist = smoke_pos.dist(target_center);
        if (smoke_to_target_dist > missile_to_target_dist)
            continue;

        // 采样底部圆周
        for (size_t i = 0; i < sample_points; ++i)
        {
            const double angle = angle_step * i;
            const Point bottom_point(
                bottom_center.x + target_radius * cos(angle),
                bottom_center.y + target_radius * sin(angle),
                bottom_center.z);

            // 计算烟幕到导弹-底部点连线的距离
            const double dist_to_line = smoke_pos.dist2line(missile_pos, bottom_point);
            if (dist_to_line > smoke_radius)
                return false;
        }

        // 采样顶部圆周
        for (size_t i = 0; i < sample_points; ++i)
        {
            const double angle = angle_step * i;
            const Point top_point(
                top_center.x + target_radius * cos(angle),
                top_center.y + target_radius * sin(angle),
                top_center.z);

            // 计算烟幕到导弹-顶部点连线的距离
            double dist_to_line = smoke_pos.dist2line(missile_pos, top_point);
            if (dist_to_line > smoke_radius)
                return false;
        }
        blocked_smoke++;
    }

    return active_smoke && blocked_smoke;
}

struct SimulationResult
{
    std::vector<std::pair<double, double>> blocked_intervals;
    double total_blocked_time;
};

std::vector<SimulationResult> simulate_cpp(
    const std::vector<Missile> &missiles,
    const std::vector<Smoke> &smokes,
    double time,
    double step,
    bool debug)
{
    std::vector<SimulationResult> results(missiles.size());

    double current_time = 0.0;
    std::vector<std::vector<bool>> blocked_status(missiles.size());
    std::vector<std::vector<double>> time_points(missiles.size());

    double stime = 0.0;
    for (const auto &s : smokes)
        if (s.start_time > stime)
            stime = s.start_time;
    time = std::max(time, stime + 21.0);

    while (current_time <= time)
    {
        for (size_t i = 0; i < missiles.size(); ++i)
        {
            const bool is_blocked = missiles[i].is_blocked(current_time, smokes);
            blocked_status[i].push_back(is_blocked);
            time_points[i].push_back(current_time);
        }
        current_time += step;
    }

    if (debug)
    {
        for (size_t i = 0; i < missiles.size(); ++i)
        {
            std::ofstream ofs("missile_" + std::to_string(i) + "_debug.txt");
            for (size_t j = 0; j < blocked_status[i].size(); ++j)
            {
                const Point pos = missiles[i].getpos(time_points[i][j]);
                ofs << "Time:\t" << time_points[i][j]
                    << "\nPos:\t(" << pos.x << "," << pos.y << "," << pos.z
                    << ")\tBlocked:\t" << blocked_status[i][j] << "\n";
            }
            ofs.close();
        }
    }

    // 计算遮挡区间
    for (size_t i = 0; i < missiles.size(); ++i)
    {
        std::vector<std::pair<double, double>> intervals;
        double current_interval_start = -1;

        for (size_t j = 0; j < blocked_status[i].size(); ++j)
        {
            if (blocked_status[i][j] && current_interval_start < 0)
            {
                current_interval_start = time_points[i][j];
            }
            else if (!blocked_status[i][j] && current_interval_start >= 0)
            {
                intervals.push_back({current_interval_start, time_points[i][j]});
                current_interval_start = -1;
            }
        }

        if (current_interval_start >= 0)
        {
            intervals.push_back({current_interval_start, time});
        }

        double total_blocked = 0.0;
        for (const auto &interval : intervals)
        {
            total_blocked += interval.second - interval.first;
        }

        results[i].blocked_intervals = intervals;
        results[i].total_blocked_time = total_blocked;
    }

    return results;
}

PYBIND11_MODULE(libsimulate, m)
{
    pybind11::class_<Point>(m, "Point")
        .def(pybind11::init<double, double, double>(),
             pybind11::arg("x") = 0, pybind11::arg("y") = 0, pybind11::arg("z") = 0)
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def_readwrite("z", &Point::z);

    pybind11::class_<Missile>(m, "Missile")
        .def(pybind11::init<const Point &>());

    pybind11::class_<Smoke>(m, "Smoke")
        .def(pybind11::init<const Point &, double>());

    pybind11::class_<SimulationResult>(m, "SimulationResult")
        .def_readwrite("blocked_intervals", &SimulationResult::blocked_intervals)
        .def_readwrite("total_blocked_time", &SimulationResult::total_blocked_time);

    m.def("simulate_cpp", &simulate_cpp);
}