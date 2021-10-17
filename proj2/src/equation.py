class Equation():
    """A class used to represent a logical statement about a free cell.
    count - total number of undiscovered adjacent blocked cells
    cells - set of unconfirmed adjacent cells
    """
    def __init__(self, cells, count):
        self.cells = cells
        self.count = count

    def __str__(self):
        ret = ""
        for cell in self.cells:
            ret += f"({cell.x},{cell.y})"
            
        return ret + " = " + str(self.count)
    
    def __repr__(self):
        return self.__str__()