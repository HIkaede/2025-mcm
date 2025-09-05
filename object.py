import math


class point:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar):
        return point(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        """向量归一化"""
        l = self.length()
        if l == 0:
            return point(0, 0, 0)
        return point(self.x / l, self.y / l, self.z / l)


class missile(point):
    def __init__(self, pos):
        super().__init__(pos.x, pos.y, pos.z)
        self.v = self.normalize() * -300


class smoke(point):
    def __init__(self, pos, start):
        super().__init__(pos.x, pos.y, pos.z)
        self.start = start


class drone(point):
    def __init__(self, pos, v):
        super().__init__(pos.x, pos.y, pos.z)
        self.v = v

    def getpos(self, t):
        return point(
            self.x + self.v.x * t,
            self.y + self.v.y * t,
            self.z,
        )

    def drop_smoke(self, t, delay):
        drop_pos = self.getpos(t + delay)
        fall = point(0, 0, -4.9 * delay**2)
        return smoke(drop_pos + fall, t + delay)
