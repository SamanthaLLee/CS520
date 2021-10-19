class Cell(object):
    """
    A class used to represent a Cell in the gridworld

    ...

    Attributes
    ----------
    x : int
        the x-coordinate of the cell
    y : int
        the y-coordinate of the cell
    g : float
         the length of the shortest path discovered from the initial search point to cell n so far
    h : float
        the heuristic value, estimating the distance from the cell n to the goal node
    f : float
        an estimate of the distance from the initial search node to the final goal node through cell n (g+h)
    blocked : boolean
        whether the cell is actually blocked in the gridworld
    seen : boolean
        blocked cells: whether the agent has seen that the current cell is blocked
        free cells: whether the agent has occupied the current cell
    parent : Cell
        a pointer to the previous node along the shortest path to n
    child : Cell
        a pointer to the next node along the shortest path to n
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.id = int
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')
        self.blocked = False
        self.seen = False
        self.parent = None
        self.child = None

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return "("+str(self.x) + "," + str(self.y) + ") | id: " + str(self.id)

    def __repr__(self):
        return self.__str__()
