#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    bool is_blocked(double t, bool check_mid, const std::vector<class Smoke> &smoke_list) const;
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

class SamplePoint
{
    std::vector<Point> tb_samples;
    std::vector<Point> full_samples;
    bool initialized;
    static SamplePoint instance;

public:
    SamplePoint() : initialized(false) {}
    static SamplePoint &get_instance()
    {
        return instance;
    }

    void initialize()
    {
        if (initialized)
            return;

        const Point bottom_center(0, 200, 0);
        const Point top_center(0, 200, 10);
        constexpr double target_radius = 7.0;
        constexpr int32_t sample_points = 36;
        constexpr double angle_step = 2.0 * M_PI / sample_points;

        tb_samples.reserve(2 * sample_points);
        full_samples.reserve(7 * sample_points);

        // 采样底部圆周
        for (size_t i = 0; i < sample_points; ++i)
        {
            const double angle = angle_step * i;
            const Point bottom_point(
                bottom_center.x + target_radius * cos(angle),
                bottom_center.y + target_radius * sin(angle),
                bottom_center.z);
            tb_samples.emplace_back(bottom_point);
            full_samples.emplace_back(bottom_point);
        }

        // 采样顶部圆周
        for (size_t i = 0; i < sample_points; ++i)
        {
            const double angle = angle_step * i;
            const Point top_point(
                top_center.x + target_radius * cos(angle),
                top_center.y + target_radius * sin(angle),
                top_center.z);
            tb_samples.emplace_back(top_point);
            full_samples.emplace_back(top_point);
        }

        // 采样中间层圆周
        constexpr int32_t vertical_layers = 5;
        for (int32_t layer = 1; layer <= vertical_layers; ++layer)
        {
            const double z_height = bottom_center.z + (top_center.z - bottom_center.z) * layer / (vertical_layers + 1);
            for (size_t i = 0; i < sample_points; ++i)
            {
                const double angle = angle_step * i;
                const Point middle_point(
                    bottom_center.x + target_radius * cos(angle),
                    bottom_center.y + target_radius * sin(angle),
                    z_height);
                full_samples.emplace_back(middle_point);
            }
        }

        initialized = true;
    }
    const std::vector<Point> &getSamples(bool check_mid) const { return check_mid ? full_samples : tb_samples; }
    size_t getTotalSampleCount(bool check_mid) const
    {
        return check_mid ? full_samples.size() : tb_samples.size();
    }
};

SamplePoint SamplePoint::instance;

bool Missile::is_blocked(double t, bool check_mid, const std::vector<Smoke> &smoke_list) const
{
    Point missile_pos = getpos(t);
    const Point target_center(0, 200, 5);
    constexpr double smoke_radius = 10.0;

    // 所有在导弹和目标之间的活跃烟幕
    std::vector<Point> active_smoke_positions;
    const double missile_to_target_dist = missile_pos.dist(target_center);

    for (const auto &smoke : smoke_list)
    {
        if (!smoke.is_active(t))
            continue;

        const Point smoke_pos = smoke.getpos(t);
        const double smoke_to_target_dist = smoke_pos.dist(target_center);

        if (smoke_pos.dist(missile_pos) <= smoke_radius)
            return true;

        if (smoke_to_target_dist <= missile_to_target_dist)
            active_smoke_positions.push_back(smoke_pos);
    }

    if (active_smoke_positions.empty())
        return false;

    SamplePoint &samples = SamplePoint::get_instance();
    samples.initialize();

    // 检查每个采样点是否被烟幕遮蔽
    size_t blocked_samples = 0;
    for (const auto &sample_point : samples.getSamples(check_mid))
    {
        bool is_sample_blocked = false;
        for (const auto &smoke_pos : active_smoke_positions)
        {
            const double dist_to_line = smoke_pos.dist2line(missile_pos, sample_point);
            if (dist_to_line <= smoke_radius)
            {
                is_sample_blocked = true;
                break;
            }
        }
        if (is_sample_blocked)
            blocked_samples++;
    }

    return blocked_samples == samples.getTotalSampleCount(check_mid);
}

struct SimulationResult
{
    std::vector<std::pair<double, double>> blocked_intervals;
    double total_blocked_time;
};

std::vector<SimulationResult> simulate_cpp(
    const std::vector<Missile> &missiles,
    const std::vector<Smoke> &smokes,
    bool check_mid,
    double step)
{
    std::vector<SimulationResult> results(missiles.size());

    double current_time = 0.0;
    std::vector<std::vector<bool>> blocked_status(missiles.size());
    std::vector<std::vector<double>> time_points(missiles.size());

    double time = 0.0;
    for (const auto &s : smokes)
        if (s.start_time > time)
            time = s.start_time;
    time += 21.0;

    while (current_time <= time)
    {
        for (size_t i = 0; i < missiles.size(); ++i)
        {
            const bool is_blocked = missiles[i].is_blocked(current_time, check_mid, smokes);
            blocked_status[i].push_back(is_blocked);
            time_points[i].push_back(current_time);
        }
        current_time += step;
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