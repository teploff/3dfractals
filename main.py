from ursina import Ursina, camera, window, Light, color, scene, Entity, held_keys, time, Mesh, Vec3, EditorCamera


class Game(Ursina):
    def __init__(self):
        super().__init__()
        window.color = color.black
        # window.fullscreen_size = 1920, 1080
        # window.fullscreen = True
        Light(type='ambient', color=(0.5, 0.5, 0.5, 1))
        Light(type='directional', color=(0.5, 0.5, 0.5, 1), direction=(1, 1, 1))
        self.MAP_SIZE = 20
        self.new_game()
        camera.position = (self.MAP_SIZE // 2, -20.5, -20)
        camera.rotation_x = -57
        self.cube = Entity(model='cube', rotation=(-15, 0, 0), position=(0, 0, 0), scale=4, color=color.orange)

        verts = (Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0),)
        tris = ((0, 1, 2, 0), )

        lines = Entity(model=Mesh(vertices=verts, mode='ngon', thickness=1), color=color.cyan)
        # points = Entity(model=Mesh(vertices=verts, mode='point', thickness=10), color=color.red, z=-1.01)

        self.triangulator = Mesh()

        EditorCamera()

    def new_game(self):
        scene.clear()

    def update(self):
        self.cube.rotation_y += time.dt * 10
        self.cube.x += held_keys['d'] * time.dt * 5
        self.cube.x -= held_keys['a'] * time.dt * 5
        self.cube.y += held_keys['w'] * time.dt * 5
        self.cube.y -= held_keys['s'] * time.dt * 5

    def input(self, key):
        if key == '2':
            camera.rotation_x = 0
            camera.position = (self.MAP_SIZE // 2, self.MAP_SIZE // 2, -50)
        elif key == '3':
            camera.position = (self.MAP_SIZE // 2, -20.5, -20)
            camera.rotation_x = -57
        super().input(key)


if __name__ == '__main__':
    game = Game()
    update = game.update
    game.run()
