from enum import IntEnum


class Terrain(IntEnum):
    FLAT = 0  # 0.2
    HILLY = 1  # 0.5
    FOREST = 2  # 0.8
    BLOCKED = 3  # 1
