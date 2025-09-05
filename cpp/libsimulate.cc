#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

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

Point farthest_point_on_circle(const Point &center, double radius, const Point &normal,
                               const Point &line_point, const Point &line_dir)
{
    Point pc = center - line_point;
    Point line_dir_normalized = line_dir.normalize();
    double t = pc.dot(line_dir_normalized);
    Point closest_on_line = line_point + line_dir_normalized * t;

    Point away_from_line = center - closest_on_line;

    if (away_from_line.length() < 1e-10)
    {
        Point temp = (std::abs(line_dir_normalized.x) < 0.9) ? Point(1, 0, 0) : Point(0, 1, 0);
        away_from_line = line_dir_normalized.cross(temp);
    }

    Point normal_normalized = normal.normalize();
    double projection_on_normal = away_from_line.dot(normal_normalized);
    Point away_in_plane = away_from_line - normal_normalized * projection_on_normal;

    if (away_in_plane.length() < 1e-10)
    {
        Point temp = (std::abs(normal_normalized.x) < 0.9) ? Point(1, 0, 0) : Point(0, 1, 0);
        away_in_plane = normal_normalized.cross(temp);
    }

    Point direction = away_in_plane.normalize();
    return center + direction * radius;
}

bool Missile::is_blocked(double t, const std::vector<Smoke> &smoke_list) const
{
    Point missile_pos = getpos(t);
    Point target_center(0, 200, 5);
    double target_radius = 7.0;

    Point bottom_center(0, 200, 0);
    Point top_center(0, 200, 10);
    Point normal_z(0, 0, 1);

    for (const auto &smoke : smoke_list)
    {
        if (!smoke.is_active(t))
            continue;

        Point smoke_pos = smoke.getpos(t);

        double missile_to_target_dist = missile_pos.dist(target_center);
        double smoke_to_target_dist = smoke_pos.dist(target_center);
        if (smoke_to_target_dist > missile_to_target_dist)
            continue;

        // 检查底部圆
        Point missile_to_smoke = smoke_pos - missile_pos;
        Point farthest_bottom = farthest_point_on_circle(
            bottom_center, target_radius, normal_z, missile_pos, missile_to_smoke);
        double dist_bottom = smoke_pos.dist2line(missile_pos, farthest_bottom);
        bool bottom_blocked = dist_bottom <= 10.0;

        // 检查顶部圆
        Point farthest_top = farthest_point_on_circle(
            top_center, target_radius, normal_z, missile_pos, missile_to_smoke);
        double dist_top = smoke_pos.dist2line(missile_pos, farthest_top);
        bool top_blocked = dist_top <= 10.0;

        if (bottom_blocked && top_blocked)
            return true;
    }
    return false;
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
            bool is_blocked = missiles[i].is_blocked(current_time, smokes);
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
                Point pos = missiles[i].getpos(time_points[i][j]);
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