import numpy as np
import matplotlib.pyplot as plt
import solve
import time


def isfloat(str):
    """Determines whether a given string can be converted to float"""
    try:
        float(str)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    dim = input("What is the length of your gridworld? ")
    while not dim.isdigit() or int(dim) < 2:
        dim = input("Enter a valid length. ")

    p = input("With what probability will a cell be blocked? ")
    while not isfloat(p) or float(p) > 1 or float(p) < 0:
        p = input("Enter a valid probability. ")

    solve.generategridworld(int(dim), float(p))

    starttime = time.time()
    result = solve.solve3()
    solve.printGridworld()
    endtime = time.time()
    if (result is None):
        print("No solution.")

    solve.trajectorylen = solve.trajectorylen if result is not None else None
    print("Trajectory length:", solve.trajectorylen)
    print("Cells processed: ", solve.numcellsprocessed)
    print("Runtime: ", endtime - starttime, "s")
