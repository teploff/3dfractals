from ursina import Ursina, camera, window, Light, color, scene, Entity, held_keys, time, Mesh, Vec3, EditorCamera
from ursina.scripts.generate_normals import generate_normals
import numpy


class Game(Ursina):
    def __init__(self):
        super().__init__()
        window.color = color.black
        window.fullscreen_size = 1920, 1080
        window.fullscreen = True

        Light(type='ambient', color=(0.5, 0.5, 0.5, 1))
        Light(type='directional', color=(0.5, 0.5, 0.5, 1), direction=(1, 1, 1))

        self.new_game()
        verts = ((0, 0, 1), (3, 0, -1), (3, 3, 0.5), (0, 3, -1), (-3, 3, 1), (-3, 0, -0.5), (-3, -3, 0), (0, -3, 0), (3, -3, 0))
        tris = ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6), (0, 6, 7), (0, 7, 8), (0, 8, 1))
        norms = generate_normals(verts, triangles=tris)

        colors = (color.red, color.blue, color.lime, color.black, color.green, color.yellow, color.smoke, color.magenta)

        self.surface = Entity(model=Mesh(vertices=verts, triangles=tris, normals=norms.tolist(), mode='line', colors=colors, thickness=3), scale=2)
        self.surface.model.colorize(smooth=True)

        EditorCamera()

    @staticmethod
    def new_game():
        scene.clear()

    def update(self):
        self.surface.rotation_y += time.dt * 10
        self.surface.x += held_keys['d'] * time.dt * 5
        self.surface.x -= held_keys['a'] * time.dt * 5
        self.surface.y += held_keys['w'] * time.dt * 5
        self.surface.y -= held_keys['s'] * time.dt * 5

    def input(self, key):
        super().input(key)


if __name__ == '__main__':
    game = Game()
    update = game.update
    game.run()
