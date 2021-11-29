import math
from typing import Optional, Tuple
import uuid


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y and self.z == other.z

        return False

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.x *= other
            self.y *= other
            self.z *= other

            return self

    def __add__(self, other):
        if isinstance(other, Tuple) or isinstance(other, Tuple):
            self.x += other[0]
            self.y += other[1]
            self.z += other[2]

            return self

    def __repr__(self):
        return f'({self.x}; {self.y}; {self.z})'

    def __hash__(self):
        return id(self)


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
    def __init__(self, p1: Point, p2: Point, p3: Point, max_depth: int, parent=None, mark=False, special=False, limit=None, iters=None):
        """
        Конструктор создания грани (треугольника).
        :param p1: Первая точка грани.
        :param p2: Вторая точка грани.
        :param p3: Третья точка грани.
        :param max_depth: Максимальная глубина роста грани. Максимальная глубина = максимальной глубине тетраэдра,
        на котором эта грань расположена.
        :param parent: Родительский объект. В данном контексте это тетраэдр. Используется для корректного вычисления
        паралелльного переноса между дочерним тетраэдром, основанием которого является та самая грань, и родительским.
        :param mark: Булевская метка. Используется для того, чтобы отследить тетраэдры, которые учасвтсввуют в
        вычислении метрик общей фигуры: длины, площади и объема. Ведь производные тетраэдры берут свое начало с грани
        родительского тетраэдра (трегольника). TODO: Наверняка какой-либо пораждающий патерн или враппер над
        TODO: существующим классом позволил бы избежать данного параметра тут.
        :param limit: Предельное значение, до которого должна дорасти грань. TODO: временное решение, возможно это
        TODO: необходимо как-то обыграть в будущем
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.max_depth = max_depth

        self.parent = parent

        self.mark = mark
        self.special = special
        self.limit = limit
        self.iters = iters

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
        return (self.p3.x - self.p1.x) * (self.p2.z - self.p1.z) - (self.p2.x - self.p1.x) * (self.p3.z - self.p1.z)

    @property
    def _z(self) -> float:
        return (self.p2.x - self.p1.x) * (self.p3.y - self.p1.y) - (self.p2.y - self.p1.y) * (self.p3.x - self.p1.x)

    @property
    def total_length(self) -> float:
        return self._line1.length + self._line2.length + self._line3.length


class Tetrahedron:
    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, surface_k: Tuple[float, float, float],
                 parent: Optional[Face], mark: bool = False):
        """
        Конструктор создания тетраэдра.
        :param p1: Первая точка основания тетраэдра.
        :param p2: Вторая точка основания тетраэдра.
        :param p3: Третья точка основания тетраэдра.
        :param p4: Вершина тетраэдра.
        :param surface_k: Вектор нормали. Состоящий из трех коэффициентов: A, B и C.
        :param parent: Родительский объект. В данном контексте это грань родительского тетраэдра. Используется для
        корректного вычисления результирующей площади полной фигуры.
        :param mark: Булевская метка. Используется для того, чтобы отследить тетраэдры, которые учасвтсввуют в
        вычислении метрик общей фигуры: длины, площади и объема. TODO: Наверняка какой-либо пораждающий патерн или
        TODO: враппер над существующим классом позволил бы избежать данного параметра тут.
        """
        self._id = str(uuid.uuid4())

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

        self.A = surface_k[0]
        self.B = surface_k[1]
        self.C = surface_k[2]

        self.parent = parent

        self.mark = mark

        self._line1 = Line(p1, p4)
        self._line2 = Line(p2, p4)
        self._line3 = Line(p3, p4)
        self._line4 = Line(p1, p2)
        self._line5 = Line(p2, p3)
        self._line6 = Line(p3, p1)

        # TODO: Возможно необходимо обзавестись свойством max и current depth. Потому что тут 0 - это заглушка. Нет пока
        # TODO: возможности передавать сюда максимальную глубину грани, лежащую на текущем тетраэдре
        self._face1 = Face(p1, p4, p2, 0)
        self._face2 = Face(p2, p4, p3, 0)
        self._face3 = Face(p3, p4, p1, 0)
        self._face4 = Face(p1, p2, p3, 0)

    def __repr__(self):
        return f'Tetrahedron:\n {self._face1};\n {self._face2};\n {self._face3};\n {self._face4}'

    @property
    def id(self):
        """

        :return:
        """
        return self._id

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
        if self.parent is None:
            return self._face1.square + self._face2.square + self._face3.square + self._face4.square

        # Если существует родиетельская грань, то мы дожны учесть разницу. Поэтому вычитаем одну из.
        return self._face1.square + self._face2.square + self._face3.square - self._face4.square

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
