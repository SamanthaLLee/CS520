import math

def getManhattanDistance(x1, y1, x2, y2, weight):
    """Manhattan: d((x1, y1),(x2, y2)) = abs(x1 - x2) + abs(y1 - y2)"""
    return (abs(x1-x2) + abs(y1-y2))*weight

def getEuclideanDistance(x1, y1, x2, y2, weight):
    """Euclidean: d((x1, y1),(x2, y2)) = sqrt((x1 - x2)2 + (y1 - y2)2)"""
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)*weight

def getChebyshevDistance(x1, y1, x2, y2, weight):
    """Chebyshev: d((x1, y1),(x2, y2)) = max((x1 - x2), (y1 - y2))"""
    return max((x1 - x2), (y1 - y2))*weight
