class Equation():
    """A class used to represent a logical statement about a free cell.
    count - total number of undiscovered adjacent blocked cells
    cells - set of unconfirmed adjacent cells
    """
    def __init__(self, cell, count):
        self.cells = set()
        self.count = count

        
