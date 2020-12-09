from ursina import Ursina, camera, window, Light, color, scene, Entity, held_keys, time, Mesh, Vec3, EditorCamera
from ursina.scripts.generate_normals import generate_normals
import numpy
from scipy.spatial import Delaunay
from typing import List, Tuple


class Game(Ursina):
    def __init__(self):
        super().__init__()
        window.color = color.black
        window.fullscreen_size = 1920, 1080
        window.fullscreen = True

        Light(type='ambient', color=(0.5, 0.5, 0.5, 1))
        Light(type='directional', color=(0.5, 0.5, 0.5, 1), direction=(1, 1, 1))

        ss = Surface([
            Triangle(
                Line(Point(0.0, 0.0, 0.0), Point(3.0, 4.0, 0.0)),
                Line(Point(3.0, 4.0, 0.0), Point(-3.0, 4.0, 0.0)),
                Line(Point(-3.0, 4.0, 0.0), Point(0.0, 0.0, 0.0))
            ),
        ])
        self.new_game()

        colors = (color.red, color.blue, color.lime, color.black, color.green, color.yellow, color.smoke, color.magenta)

        for _ in range(6):
            ss.fraction()
        a1, a2, a3 = ss.get_ursina_samples()
        self.surface = Entity(
            model=Mesh(vertices=a1, triangles=a2, normals=a3, mode='line', colors=colors, thickness=3),
            scale=2)
        self.surface.model.colorize(smooth=False)

        EditorCamera()

    @staticmethod
    def new_game():
        scene.clear()

    def update(self):
        self.surface.rotation_y += held_keys['e'] * time.dt * 10
        self.surface.rotation_y -= held_keys['q'] * time.dt * 10
        self.surface.x += held_keys['d'] * time.dt * 5
        self.surface.x -= held_keys['a'] * time.dt * 5
        self.surface.y += held_keys['w'] * time.dt * 5
        self.surface.y -= held_keys['s'] * time.dt * 5

    def input(self, key):
        super().input(key)

    def get_midpoint(self):
        pass


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def midpoint(self):
        return Point((self.p1.x + self.p2.x) / 2.0, (self.p1.y + self.p2.y) / 2.0, (self.p1.z + self.p2.z) / 2.0)


class Triangle:
    def __init__(self, line1: Line, line2: Line, line3: Line):
        self.line1 = line1
        self.line2 = line2
        self.line3 = line3


class Surface:
    def __init__(self, triangles):
        self.triangles = triangles

    def get_ursina_samples(self) -> (tuple, tuple, tuple):
        vertices = []
        triangles = []

        for i, triangle in enumerate(self.triangles):
            triangles.append((i * 2 + i, i * 2 + i + 1, i * 2 + i + 2, i * 2 + i))
            vertices.append((triangle.line1.p1.x, triangle.line1.p1.y, triangle.line1.p1.z))
            vertices.append((triangle.line2.p1.x, triangle.line2.p1.y, triangle.line2.p1.z))
            vertices.append((triangle.line3.p1.x, triangle.line3.p1.y, triangle.line3.p1.z))

        return vertices, triangles, generate_normals(vertices, triangles=triangles).tolist()

    def fraction(self):
        new_triangles = []

        for triangle in self.triangles:
            p1 = triangle.line1.midpoint()
            p2 = triangle.line2.midpoint()
            p3 = triangle.line3.midpoint()

            new_triangles.append(Triangle(
                Line(Point(triangle.line1.p1.x, triangle.line1.p1.y, triangle.line1.p1.z), Point(p1.x, p1.y, p1.z)),
                Line(Point(p1.x, p1.y, p1.z), Point(p3.x, p3.y, p3.z)),
                Line(Point(p3.x, p3.y, p3.z), Point(triangle.line1.p1.x, triangle.line1.p1.y, triangle.line1.p1.z)),
            ))
            new_triangles.append(Triangle(
                Line(Point(p1.x, p1.y, p1.z), Point(triangle.line2.p1.x, triangle.line2.p1.y, triangle.line2.p1.z)),
                Line(Point(triangle.line2.p1.x, triangle.line2.p1.y, triangle.line2.p1.z), Point(p2.x, p2.y, p2.z)),
                Line(Point(p2.x, p2.y, p2.z), Point(p1.x, p1.y, p1.z)),
            ))
            new_triangles.append(Triangle(
                Line(Point(p2.x, p2.y, p2.z), Point(triangle.line3.p1.x, triangle.line3.p1.y, triangle.line3.p1.z)),
                Line(Point(triangle.line3.p1.x, triangle.line3.p1.y, triangle.line3.p1.z), Point(p3.x, p3.y, p3.z)),
                Line(Point(p3.x, p3.y, p3.z), Point(p2.x, p2.y, p2.z)),
            ))
            new_triangles.append(Triangle(
                Line(Point(p3.x, p3.y, p3.z), Point(p1.x, p1.y, p1.z)),
                Line(Point(p1.x, p1.y, p1.z), Point(p2.x, p2.y, p2.z)),
                Line(Point(p2.x, p2.y, p2.z), Point(p3.x, p3.y, p3.z)),
            ))

        self.triangles = new_triangles


if __name__ == '__main__':
    game = Game()
    update = game.update
    game.run()
