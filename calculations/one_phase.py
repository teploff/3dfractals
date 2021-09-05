import math

from entity import Point, Line, Face, Tetrahedron


MAX_DEPTH = 5
LIMIT_VALUE = 2.0

# ONE PHASE CONSTANTS
ITER_COUNT = 10


class FractalState:
    def __init__(self):
        self.lines = []
        self.triangles = []
        self.tetrahedron = []

    def append_lines(self, line: Line):
        for l in self.lines:
            # Если такая прямая уже существует в списке наших прямых, ничего не добавляем
            if (l.p1 == line.p1 and l.p2 == line.p2) or (l.p1 == line.p2 and l.p2 == line.p1):
                return
        self.lines.append(line)

    def append_triangles(self, face: Face):
        self.triangles.append(face)

    def append_tetrahedron(self, tetrahedron: Tetrahedron):
        self.tetrahedron.append(tetrahedron)

    @property
    def calc_line_length(self):
        length = 0

        for l in self.lines:
            length += l.length

        return length

    @property
    def calc_square(self):
        square = 0

        for t in self.triangles:
            square += t.square

        return square

    @property
    def calc_volume(self):
        volume = 0

        for t in self.tetrahedron:
            volume += t.volume

        return volume


def growth_triangle(p1: Point, p2: Point, p3: Point, coefficient: float) -> (Point, Point, Point):
    new_p1 = Point(p1.x * coefficient, p1.y * coefficient, p1.z * coefficient)
    new_p2 = Point(p2.x * coefficient, p2.y * coefficient, p2.z * coefficient)
    new_p3 = Point(p3.x * coefficient, p3.y * coefficient, p3.z * coefficient)

    return new_p1, new_p2, new_p3


def calculate(iter_count: int, limit_value: float):
    # начальные точки
    s_p1 = Point(0.0, 0.0, 0.0)
    s_p2 = Point(0.5, (math.sqrt(3) / 2.0), 0.0)
    s_p3 = Point(1.0, 0.0, 0.0)

    # Состояние фрактала на каждой его из итераций
    fractal_states = []

    # Определяем коэффициент
    coefficient = limit_value / float(iter_count)

    # Растем треугольник
    for i in range(iter_count):
        fractal_state = FractalState()
        c = coefficient + coefficient * i

        p1, p2, p3 = growth_triangle(s_p1, s_p2, s_p3, c)

        lines = [Line(p1, p2), Line(p2, p3), Line(p1, p3)]
        for l in lines:
            fractal_state.append_lines(l)

        fractal_state.append_triangles(Face(p1, p2, p3))

        fractal_states.append(fractal_state)

    for iteration, state in enumerate(fractal_states):
        print(f'state = {iteration}. Total line length = {state.calc_line_length}. Total square = {state.calc_square}. Total volume = {state.calc_volume}')


if __name__ == '__main__':
    calculate(ITER_COUNT, LIMIT_VALUE)
