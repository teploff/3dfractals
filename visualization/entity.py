from typing import List

from ursina import Entity, color
from ursina import Mesh


class Model:
    def __init__(self, vertices: List[List[float]], triangles: List[List[int]]):
        self.vertices = vertices
        self.triangles = triangles


class Builder:
    def __init__(self, sequence):
        self.sequence = sequence

    def append_material(self, material: List[Model]):
        self.sequence.append(material)

    def gen(self, item: int):
        if item >= len(self.sequence):
            raise KeyError

        model = self.sequence[item][0]
        parent_entity = Entity(model=Mesh(vertices=model.vertices, triangles=model.triangles, thickness=4,
                                          mode='line'), scale=2, color=color.yellow)

        if len(self.sequence[item]) == 1:
            return parent_entity

        for model in self.sequence[item][1:]:
            Entity(model=Mesh(vertices=model.vertices, triangles=model.triangles, thickness=4, mode='line'),
                   scale=2, color=color.yellow)

        return parent_entity
