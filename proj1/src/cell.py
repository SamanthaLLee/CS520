class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')
        self.blocked = False
        self.seen = False
        self.parent = None
        self.child = None
