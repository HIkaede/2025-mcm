import math


class point:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return point(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return point(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def dist(self, other):
        return (
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        ) ** 0.5

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        l = self.length()
        if l == 0:
            return point(0, 0, 0)
        return point(self.x / l, self.y / l, self.z / l)

    def dist2line(self, a, b):
        ab = b - a
        ap = self - a
        ab2 = ab.dot(ab)
        t = ap.dot(ab) / ab2
        proj = a + ab * t
        return self.dist(proj)

    def farthest_point_on_circle(self, center, radius, normal, line_point, line_dir):
        """找到圆周上距离给定直线最远的点"""
        # 1. 计算直线到圆心的最近点
        pc = center - line_point
        line_dir_normalized = line_dir.normalize()
        t = pc.dot(line_dir_normalized)
        closest_on_line = line_point + line_dir_normalized * t

        # 2. 从最近点指向圆心的向量（远离直线方向）
        away_from_line = center - closest_on_line

        # 3. 如果圆心在直线上，选择任意垂直方向
        if away_from_line.length() < 1e-10:
            temp = (
                point(1, 0, 0) if abs(line_dir_normalized.x) < 0.9 else point(0, 1, 0)
            )
            away_from_line = line_dir_normalized.cross(temp)

        # 4. 将远离向量投影到圆所在平面
        normal_normalized = normal.normalize()
        projection_on_normal = away_from_line.dot(normal_normalized)
        away_in_plane = away_from_line - normal_normalized * projection_on_normal

        # 5. 如果投影后向量为零，在圆平面内选择任意方向
        if away_in_plane.length() < 1e-10:
            temp = point(1, 0, 0) if abs(normal_normalized.x) < 0.9 else point(0, 1, 0)
            away_in_plane = normal_normalized.cross(temp)

        # 6. 归一化并缩放到圆周
        direction = away_in_plane.normalize()
        farthest_point = center + direction * radius

        return farthest_point


class missile(object):
    def __init__(self, pos):
        self.pos = pos
        self.v = pos.normalize() * -300

    def getpos(self, t):
        return point(
            self.pos.x + self.v.x * t,
            self.pos.y + self.v.y * t,
            self.pos.z + self.v.z * t,
        )

    def is_blocked(self, t, smoke_list):
        """判断导弹在时刻t是否被烟幕遮挡"""
        missile_pos = self.getpos(t)
        target_center = point(0, 200, 5)  # 真目标中心
        target_radius = 7.0  # 目标半径

        bottom_center = point(0, 200, 0)
        top_center = point(0, 200, 10)
        normal_z = point(0, 0, 1)

        for smoke in smoke_list:
            if not smoke.is_active(t):
                continue

            smoke_pos = smoke.getpos(t)
            if smoke_pos is None:
                continue

            # 检查烟幕是否在导弹和目标之间
            missile_to_target_dist = missile_pos.dist(target_center)
            smoke_to_target_dist = smoke_pos.dist(target_center)

            # 烟幕必须在导弹和目标之间
            if smoke_to_target_dist > missile_to_target_dist:
                continue

            # 检查上下底面是否被遮蔽
            bottom_blocked = self._is_circle_blocked(
                missile_pos, smoke_pos, bottom_center, target_radius, normal_z
            )
            top_blocked = self._is_circle_blocked(
                missile_pos, smoke_pos, top_center, target_radius, normal_z
            )

            # 如果上下底面都被遮蔽，认为目标被完全遮蔽
            if bottom_blocked and top_blocked:
                return True

        return False

    def _is_circle_blocked(
        self, missile_pos, smoke_pos, circle_center, circle_radius, normal
    ):
        """使用最远点算法判断圆是否被烟幕遮蔽"""
        # 导弹到烟幕的连线方向
        missile_to_smoke = smoke_pos - missile_pos

        # 在圆周上找到距离导弹-烟幕连线最远的点
        farthest_point = missile_pos.farthest_point_on_circle(
            circle_center, circle_radius, normal, missile_pos, missile_to_smoke
        )

        # 计算最远点到导弹-烟幕连线的距离
        dist_to_line = smoke_pos.dist2line(missile_pos, farthest_point)

        # 如果距离小于烟幕半径，则被遮蔽
        smoke_radius = 10.0
        return dist_to_line <= smoke_radius


class smoke(object):
    def __init__(self, pos, start):
        self.pos = pos
        self.start = start

    def is_active(self, t):
        return t - self.start <= 20 and t >= self.start

    def getpos(self, t):
        if not self.is_active(t):
            return None
        return point(self.pos.x, self.pos.y, self.pos.z - 3 * (t - self.start))


class drone(object):
    def __init__(self, pos, v):
        self.pos = pos
        self.v = v

    def getpos(self, t):
        return point(
            self.pos.x + self.v.x * t,
            self.pos.y + self.v.y * t,
            self.pos.z,
        )

    def drop_smoke(self, t, delay):
        drop_pos = self.getpos(t + delay)
        fall = point(0, 0, -4.9 * delay**2)
        return smoke(drop_pos + fall, t + delay)
