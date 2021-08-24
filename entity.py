import math
from typing import Optional


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f'({self.x}; {self.y}; {self.z})'


class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def __repr__(self):
        return f'Line: {self.p1}---{self.p2}'

    @property
    def length(self):
        return math.sqrt((self.p2.x - self.p1.x) ** 2 + (self.p2.y - self.p1.y) ** 2 + (self.p2.z - self.p1.z) ** 2)


class Face:
    def __init__(self, p1: Point, p2: Point, p3: Point):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self._line1 = Line(p1, p2)
        self._line2 = Line(p2, p3)
        self._line3 = Line(p3, p1)

    def __repr__(self):
        return f'Face: {self._line1}; {self._line2}; {self._line3}'

    @property
    def square(self) -> float:
        return 0.5 * math.sqrt(self._x ** 2 + self._y ** 2 + self._z ** 2)

    @property
    def _x(self) -> float:
        return (self.p2.y - self.p1.y) * (self.p3.z - self.p1.z) - (self.p3.y - self.p1.y) * (self.p2.z - self.p1.z)

    @property
    def _y(self) -> float:
        return (self.p3.x - self.p1.x) * (self.p2.z - self.p1.z) - (self.p2.x - self.p1.x) * (self.p2.z - self.p1.z)

    @property
    def _z(self) -> float:
        return (self.p2.x - self.p1.x) * (self.p3.y - self.p1.y) - (self.p2.y - self.p1.y) * (self.p3.x - self.p1.x)

    @property
    def total_length(self) -> float:
        return self._line1.length + self._line2.length + self._line3.length


class Tetrahedron:
    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, parent: Optional[Face]):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

        self.parent = parent

        self._line1 = Line(p1, p4)
        self._line2 = Line(p2, p4)
        self._line3 = Line(p3, p4)
        self._line4 = Line(p1, p2)
        self._line5 = Line(p2, p3)
        self._line6 = Line(p3, p1)

        self._face1 = Face(p1, p4, p2)
        self._face2 = Face(p2, p4, p3)
        self._face3 = Face(p3, p4, p1)
        self._face4 = Face(p1, p2, p3)

    def __repr__(self):
        return f'Tetrahedron:\n {self._face1};\n {self._face2};\n {self._face3};\n {self._face4}'

    @property
    def total_length(self) -> float:
        """
        Calculate all line's length which are in the Tetrahedron.
        :return: total tetrahedrom's length.
        """
        return self._line1.length + self._line2.length + self._line3.length + self._line4.length + self._line5.length \
               + self._line6.length

    @property
    def total_square(self) -> float:
        """
        Calculate all face's square which are in the Tetrahedron.
        Pay attention that we don't need considering square which are on the parent face.
        :return: total tetrahedrom's square.
        """
        # TODO: need take away square in the parent surface.
        return self._face1.square + self._face2.square + self._face3.square + self._face4.square

    @property
    def volume(self):
        """
        Calculate Tetrahedron's volume.
        :return: Tetrahedron's volume.
        """
        return 1.0 / 6.0 * abs(
            (self.p2.x - self.p1.x) * (self.p3.y - self.p1.y) * (self.p4.z - self.p1.z) +
            (self.p4.x - self.p1.x) * (self.p2.y - self.p1.y) * (self.p3.z - self.p1.z) +
            (self.p3.x - self.p1.x) * (self.p4.x - self.p1.y) * (self.p2.z - self.p1.z) -
            (self.p4.x - self.p1.x) * (self.p3.y - self.p1.y) * (self.p2.z - self.p1.z) -
            (self.p2.x - self.p1.x) * (self.p4.y - self.p1.y) * (self.p3.z - self.p1.z) +
            (self.p3.x - self.p1.x) * (self.p2.y - self.p1.y) * (self.p4.z - self.p1.z)
        )
