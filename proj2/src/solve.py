import random
from array import *
from queue import PriorityQueue
from cell import Cell
from equation import Equation
import itertools

# Global gridworld of Cell objects
gridworld = []

# Global goal cell
goal = None

# Vectors that represent directions
cardinaldirections = [(1, 0), (-1, 0), (0, 1), (0, -1)]
alldirections = [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (-1, -1), (1, 1), (1, -1), (-1, 1)]

numcellsprocessed = 0
trajectorylen = 0
dim = 0
finaldiscovered = False
fullgridworld = False

equation_KB = set()


def generategridworld(d, p):
    """Generates a random gridworld based on user inputs"""
    global goal, gridworld, dim
    dim = d
    # Cells are constructed in the following way:
    # Cell(g, h, f, blocked, seen, parent)
    gridworld = [[Cell(x, y) for y in range(dim)] for x in range(dim)]
    id = 0

    # Let each cell independently be blocked with probability p, and empty with probability 1−p.
    for i in range(dim):
        for j in range(dim):
            curr = gridworld[i][j]

            # Determined block status
            rand = random.random()
            if rand < p:
                curr.blocked = 1

            # Assign ID
            curr.id = id
            id += 1

            # Set N and H
            curr.N = get_num_neighbors(i, j)
            curr.H = curr.N

    # Set the goal node
    goal = gridworld[dim-1][dim-1]

    # Ensure that the start and end positions are unblocked
    gridworld[0][0].blocked = 0
    goal.blocked = 0

    # Initialize starting cell values
    gridworld[0][0].g = 1
    gridworld[0][0].h = get_weighted_manhattan_distance(0, 0, goal.x, goal.y)
    gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
    gridworld[0][0].seen = True



def generategridworld2():
    """Generates a random gridworld based on user inputs"""
    global goal, gridworld, dim
    dim = 4
    # Cells are constructed in the following way:
    # Cell(g, h, f, blocked, seen, parent)
    gridworld = [[Cell(x, y) for y in range(dim)] for x in range(dim)]
    id = 0

    # Let each cell independently be blocked with probability p, and empty with probability 1−p.
    for i in range(dim):
        for j in range(dim):
            curr = gridworld[i][j]

            # Assign ID
            curr.id = id
            id += 1

            # Set N and H
            curr.N = get_num_neighbors(i, j)
            curr.H = curr.N

    gridworld[0][3].blocked = True
    gridworld[1][0].blocked = True
    gridworld[1][2].blocked = True
    

    # Set the goal node
    goal = gridworld[dim-1][dim-1]

    # Ensure that the start and end positions are unblocked
    gridworld[0][0].blocked = 0
    goal.blocked = 0

    # Initialize starting cell values
    gridworld[0][0].g = 1
    gridworld[0][0].h = get_weighted_manhattan_distance(0, 0, goal.x, goal.y)
    gridworld[0][0].f = gridworld[0][0].g + gridworld[0][0].h
    gridworld[0][0].seen = True




def printGridworld():
    """Prints out the current state of the gridworld.
    Key:
            B: blocked and seen
            b: blocked and unseen
            *: current path
            ' ': free space
    """
    global gridworld
    leng = len(gridworld)

    string = ''
    for i in range(leng):
        string += ('-'*(leng*2+1) + '\n')
        for j in range(leng):
            string += '|'
            curr = gridworld[i][j]
            if curr.blocked and curr.seen:
                string += 'B'
            elif curr.blocked and not curr.seen:
                string += 'b'
            elif not curr.blocked and curr.seen:
                string += '*'
            else:
                string += ' '
        string += '|\n'
    string += ('-'*(leng*2+1))
    print(string)


def astar(start, agent):
    """Performs the A* algorithm on the gridworld
    Args:
        start (Cell): The cell from which A* will find a path to the goal
    Returns:
        Cell: The head of a Cell linked list containing the shortest path
        int: Length of returned final path
    """
    global goal, gridworld, finaldiscovered, fullgridworld, cardinaldirections, numcellsprocessed
    fringe = PriorityQueue()
    fringeSet = set()
    seenSet = set()
    astarlen = 0
    goalfound = False

    curr = start
    fringe.put((curr.f, curr))
    fringeSet.add(curr.id)

    # Generate all valid children and add to fringe
    # Terminate loop if fringe is empty or if path has reached goal
    while len(fringeSet) != 0:
        f, curr = fringe.get()
        if curr is goal:
            goalfound = True
            break
        if curr.id not in fringeSet:
            continue
        fringeSet.remove(curr.id)
        seenSet.add(curr.id)
        numcellsprocessed += 1
        # if fullgridworld:
        #     print("fullgridworld", curr.x, curr.y)
        # if finaldiscovered:
        #     print("finaldiscovered", curr.x, curr.y)
        for x, y in cardinaldirections:
            xx = curr.x + x
            yy = curr.y + y

            if is_in_bounds([xx, yy]):
                nextCell = gridworld[xx][yy]
                validchild = ((agent == 1 or agent == 2) and not (nextCell.blocked and nextCell.seen)) or (
                    (agent == 3 or agent == 4) and not (nextCell.blocked and nextCell.confirmed))
                # Add children to fringe if inbounds AND unblocked and unseen
                if (finaldiscovered and nextCell.seen) or not finaldiscovered:
                    if validchild or (fullgridworld and not nextCell.blocked):
                        # Add child if not already in fringe
                        # If in fringe, update child in fringe if old g value > new g value
                        if(((not nextCell.id in fringeSet) or (nextCell.g > curr.g + 1)) and nextCell.id not in seenSet):
                            nextCell.parent = curr
                            nextCell.g = curr.g + 1
                            nextCell.h = get_weighted_manhattan_distance(
                                xx, yy, goal.x, goal.y)
                            nextCell.f = nextCell.g + nextCell.h
                            fringe.put((nextCell.f, nextCell))
                            fringeSet.add(nextCell.id)

    # Return None if no solution exists
    if len(fringeSet) == 0:
        return None, 0

    # Starting from goal cell, work backwards and reassign child attributes correctly
    if goalfound:
        parentPtr = goal
        childPtr = None
        oldParent = start.parent
        start.parent = None
        while(parentPtr is not None):
            astarlen += 1
            parentPtr.child = childPtr
            childPtr = parentPtr
            parentPtr = parentPtr.parent
        start.parent = oldParent

        return start, astarlen
    else:
        return None, 0


def solve1():
    """
    Agent 1 - The Blindfolded Agent: bumps into walls
    """
    global gridworld, cardinaldirections, trajectorylen

    agent = 1

    path, len = astar(gridworld[0][0], agent)

    # Initial A* failed - unsolvable gridworld
    if path is None:
        return None

    # Attempt to move along planned path
    curr = path
    while True:
        # A* failed - unsolvable gridworld
        if curr is None:
            return None

        # Pre-process cell
        curr.seen = True
        trajectorylen += 1

        # Goal found
        if curr.child is None:
            return path

        # Run into blocked cell
        if curr.blocked == True:
            trajectorylen -= 1
            path, len = astar(curr.parent, agent)
            curr = path

        # Continue along A* path
        else:
            # Move onto next cell along A* path
            curr = curr.child


def solve2():
    """
    Agent 2 - 4-Neighbor Agent: Can see all 4 cardinal neighbors at once
    """
    global gridworld, cardinaldirections, trajectorylen

    agent = 2

    path, len = astar(gridworld[0][0], agent)

    # Initial A* failed - unsolvable gridworld
    if path is None:
        return None

    # Attempt to move along planned path
    curr = path
    while True:
        # A* failed - unsolvable gridworld
        if curr is None:
            return None

        # Pre-process cell
        curr.seen = True
        trajectorylen += 1

        # Goal found
        if curr.child is None:
            return path

        # Run into blocked cell
        if curr.blocked == True:
            trajectorylen -= 1
            path, len = astar(curr.parent, agent)
            curr = path

        # Continue along A* path
        else:
            # Take note of environment within viewing distance (adjacent cells)
            for dx, dy in cardinaldirections:
                xx, yy = curr.x + dx, curr.y + dy

                # Only mark blocked neighbors as seen
                if is_in_bounds([xx, yy]) and gridworld[xx][yy].blocked:
                    neighbor = gridworld[xx][yy]
                    neighbor.seen = True
            # Move onto next cell along A* path
            curr = curr.child


def solve3():
    """
    Agent 3 - Example Inference Agent
    """
    global goal, gridworld, alldirections, trajectorylen

    agent = 3

    path, len = astar(gridworld[0][0], agent)

    if path is None:
        return None

    # Traverse through planned path
    curr = path
    while True:

        if(curr is None):
            return None

        # Pre-process current cell
        curr.seen = True
        curr.confirmed = True
        trajectorylen += 1

        # Goal found
        if curr.child is None:
            return path

        # Update neighbors and cascade inferences
        updatekb3(curr)

        # Replan if agent has run into blocked cell
        if curr.blocked == True:
            # print("replan cause run into block")
            trajectorylen -= 1
            curr, len = astar(curr.parent, agent)
            continue
        else:
            # Sense number of blocked and confirmed neighbors for curr
            senseorcount3(curr, True)
            # Make inferences from this sensing
            infer3(curr)

            # Replan if agent finds inferred block in path
            ptr = curr.child
            replanned = False
            while ptr.child is not None:
                if ptr.confirmed and ptr.blocked:
                    # print("replan cause inferred block")
                    curr, len = astar(curr, agent)
                    trajectorylen -= 1
                    replanned = True
                    break
                ptr = ptr.child

            # Otherwise, continue along A* path
            if not replanned:
                curr = curr.child
                # print("continue along path")


def senseorcount3(curr, sense):
    # print("sense", curr.x, curr.y)
    """Sets curr's C, E, H, B values based on current KB
    Args:
        curr (cell): current cell
    """

    if curr.blocked:
        return

    if sense:
        curr.C = 0
    curr.E = 0
    curr.B = 0
    curr.H = get_num_neighbors(curr.x, curr.y)

    for x, y in alldirections:
        xx = curr.x + x
        yy = curr.y + y

        if is_in_bounds([xx, yy]):
            neighbor = gridworld[xx][yy]
            # Only sense if agent is in curr
            if neighbor.blocked and sense:
                curr.C += 1
            # Count up all confirmed neighbors
            if neighbor.confirmed:
                if neighbor.blocked:
                    curr.B += 1
                    curr.H -= 1
                else:
                    curr.E += 1
                    curr.H -= 1


def infer3(curr):
    """Tests for the 3 given inferences
    Args:
        curr (cell): current cell to make inferences on
    """

    inferencemade = False
    # print("infer", curr.x, curr.y)
    if curr.H > 0:
        # More inferences possible on unconfirmed neighboring cells
        if curr.C == curr.B:
            inferencemade = True
            # print("curr.C == curr.B")
            # All remaining hidden neighbors are empty
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]):
                    if not gridworld[xx][yy].confirmed:
                        gridworld[xx][yy].confirmed = True
                        curr.E += 1
                        curr.H -= 1
        elif curr.N - curr.C == curr.E:
            inferencemade = True
            # print("curr.N - curr.C == curr.E")
            # All remaining hidden neighbors are blocked
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]):
                    if not gridworld[xx][yy].confirmed:
                        gridworld[xx][yy].confirmed = True
                        curr.B += 1
                        curr.H -= 1
        return inferencemade


def updatekb3(curr):
    for x, y in alldirections:
        xx = curr.x + x
        yy = curr.y + y
        if is_in_bounds([xx, yy]):
            # Find all inbounds neighbors of curr
            neighbor = gridworld[xx][yy]
            senseorcount3(neighbor, False)
            if neighbor.seen and neighbor.blocked == 0 and neighbor.H > 0:
                if infer3(neighbor):
                    for x, y in alldirections:
                        xx2 = neighbor.x + x
                        yy2 = neighbor.y + y
                        if is_in_bounds([xx2, yy2]):
                            updatekb3(gridworld[xx2][yy2])


def solve4():
    """Agent 3 - Example Inference Agent
    """

    global goal, gridworld, equation_KB, alldirections, trajectorylen

    agent = 3

    path, len = astar(gridworld[0][0], agent)

    if path is None:
        return None

    # Traverse through planned path
    curr = path
    while True:
        if(curr is None):
            return None

        # Pre-process current cell
        curr.seen = True
        curr.confirmed = True
        trajectorylen = trajectorylen + 1
        # print(f"At: {curr.x}, {curr.y}")
        # printGridworld()

        # Goal found
        if curr.child is None:
            # print(equation_KB)
            return path

        # Make basic inferences, remove cell from all eq's in KB, infer, and replan
        if curr.blocked == True:
            
            # Update counts of neighbors that have been confirmed and sensed
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]) and gridworld[xx][yy].confirmed \
                    and gridworld[xx][yy].sensed:
                    gridworld[xx][yy].H -= 1
                    gridworld[xx][yy].B += 1

            # Attempt to make basic inferences on blocked cell's neighbors
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]) and gridworld[xx][yy].confirmed \
                    and gridworld[xx][yy].sensed:
                    basic_infer(gridworld[xx][yy])
            

            # Remove blocked cell from KB
            # Attempt to make KB inferences on blocked cell's neighbors
            remove_from_KB(curr)
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]) and not gridworld[xx][yy].confirmed:
                    KB_infer(gridworld[xx][yy])

            # Backstep and replan
            trajectorylen = trajectorylen - 2
            curr, len = astar(curr.parent, agent)
            
        # Sense new cell, basic infer, add new equation to KB, 
        # remove cell from eq's in KB, infer, and replan/continue
        else:
            # Agent senses and sets curr cell's values
            # Update counts of neighbors that have been confirmed and sensed
            basic_sense(curr)
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]) and gridworld[xx][yy].confirmed \
                    and gridworld[xx][yy].sensed:
                    gridworld[xx][yy].H -= 1
                    gridworld[xx][yy].E += 1

            # Attempt to make basic inferences on free cell and cell's neighbors
            basic_infer(curr)
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]) and gridworld[xx][yy].confirmed \
                    and gridworld[xx][yy].sensed:
                    basic_infer(gridworld[xx][yy])

            # Add new equation to KB
            # Remove free cell from KB
            # Attempt to make KB inferences on blocked cell's neighbors
            add_eq_to_KB(curr)
            remove_from_KB(curr)
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]) and not gridworld[xx][yy].confirmed:
                    KB_infer(gridworld[xx][yy])

            # Replan if agent finds inferred block in path
            ptr = curr.child
            replanned = False
            while ptr.child is not None:
                if ptr.confirmed and ptr.blocked:
                    # print("replan cause inferred block")
                    curr, len = astar(curr, agent)
                    replanned = True
                    break
                ptr = ptr.child

            # Otherwise, continue along A* path
            if not replanned:
                curr = curr.child


def basic_sense(curr:Cell):
    """Sets curr's C, E, H, B, sensed values
    Args:
        curr (cell): current cell
    """

    if curr.blocked:
        return

    curr.C = 0
    curr.E = 0
    curr.B = 0
    curr.H = 0
    curr.sensed = True
    
    for x, y in alldirections:
        xx = curr.x + x
        yy = curr.y + y

        if is_in_bounds([xx, yy]):
            neighbor = gridworld[xx][yy]
            # Counts number of adjacent blocked cells (minesweeper number)
            if neighbor.blocked:
                curr.C += 1
            if neighbor.confirmed:  # Counts confirmed neighbors and if blocked/free
                curr.H -= 1
                if neighbor.blocked:
                    curr.B += 1
                else:
                    curr.E += 1
            else:  # Counts unconfirmed neighbors
                curr.H += 1


def basic_infer(curr:Cell):
    """Tests for the 3 given inferences
    If inference found:
        remove from cell from eq's in KB
        CANNOT add new equation to KB for newly free inferred cells without sensing
        infer on inferred cells' unconfirmed neighbors
    
    Args:
        curr (Cell): current cell to make inferences on
    """
    # Only infer on sensed cells with unconfirmed neighbors
    # Cannot make basic inferences on a cell without sensing that cell
    if curr.H > 0 and curr.sensed:
        # More inferences possible on unconfirmed neighboring cells
        if curr.C == curr.B:
            # print(f"{curr}: C==B")
            # All remaining hidden neighbors are empty
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]):
                    neighbor = gridworld[xx][yy]
                    if neighbor.confirmed == False:
                        # print(f"New inference: ({neighbor.x},{neighbor.y})={neighbor.blocked}")
                        neighbor.confirmed = True
                        remove_from_KB(neighbor)
                        curr.E += 1
                        curr.H -= 1
                        basic_infer_recurse_on_neighbors(neighbor)
        elif curr.N - curr.C == curr.E:
            # print(f"{curr}: N-C==E")
            # All remaining hidden neighbors are blocked
            for x, y in alldirections:
                xx = curr.x + x
                yy = curr.y + y
                if is_in_bounds([xx, yy]):
                    neighbor = gridworld[xx][yy]
                    if neighbor.confirmed == False:
                        # print(f"New inference: ({neighbor.x},{neighbor.y})={neighbor.blocked}")
                        neighbor.confirmed = True
                        remove_from_KB(neighbor)
                        curr.B += 1
                        curr.H -= 1
                        basic_infer_recurse_on_neighbors(neighbor)

def basic_infer_recurse_on_neighbors(curr):
    global alldirections
    for x, y in alldirections:
        xx = curr.x + x
        yy = curr.y + y
        if is_in_bounds([xx, yy]) and gridworld[xx][yy].sensed:
            basic_infer(gridworld[xx][yy])

def add_eq_to_KB(cell: Cell):
    """Adds the given free cell to the equation knowledge base.
    DOES NOT UPDATE KB ITSELF

    Args:
        cell (cell): given free cell
    """

    global equation_KB, alldirections

    # Adds cell's unconfirmed neighbors to new set
    unconfirmed_neighbors_set = set()
    for x, y in alldirections:
        xx = cell.x + x
        yy = cell.y + y
        if is_in_bounds([xx, yy]) and gridworld[xx][yy].confirmed == False:
            unconfirmed_neighbors_set.add(gridworld[xx][yy])

    if len(unconfirmed_neighbors_set) > 0:
        # Add new equation to KB
        new_eq = Equation(unconfirmed_neighbors_set, cell.C - cell.B)
        
        # print(f"Adding eq to KB: {new_eq}")
        equation_KB.add(new_eq)
        
        # print(f"New KB: {equation_KB}")

def remove_from_KB(cell: Cell):
    """Removes the given cell from the equation knowledge base.
    DOES NOT UPDATE KB ITSELF

    Args:
        cell (cell): given cell
    """
    global equation_KB, alldirections
    
    # Removes cell from equation's cell list 
    # and decrements equation count if cell is blocked
    print_toggle = False
    for equation in equation_KB.copy():
        if cell in equation.cells:
            print_toggle = True
            # print(f"Removing {cell} from {equation}")
            equation.cells.remove(cell)
            if cell.blocked:
                equation.count -= 1
            
            # Cleaning up the KB - remove extra equations from KB
            # 1 length equations - set cell = count
            if len(equation.cells) == 1 and (equation.count==0 or equation.count==1):
                last_cell = equation.cells.pop()
                # if last_cell.blocked == equation.count:
                #     print("good")
                # else:
                #     print("SOMETHING'S WRONG I CAN FEEL IT")
                equation_KB.remove(equation)
            # 0 length - remove from KB
            elif len(equation.cells) == 0:
                equation_KB.remove(equation)
    # if print_toggle:
    #     print(f"New KB: {equation_KB}")


def KB_infer(cell: Cell):
    """Attempts to make inferences about the given unconfirmed cell from the KB

    Args:
        cell (cell): given unconfirmed cell
    """

    global equation_KB, alldirections

    # Cannot infer anymore about this cell
    if cell.H == 0:
        return

    # Add equations from that contain cell to eqs_with_cell
    # Collect all unknown cells from equations containing cell in symbols
    total_cells = []
    eqs_with_cell = []
    for equation in equation_KB:
        if cell in equation.cells:
            total_cells.extend(list(equation.cells))
            eqs_with_cell.append(equation)
    total_cells = list(set(total_cells))

    # Generate all free/blocked combos of total_cells
    # all_combos = [tuple[Cell, bool]]
    all_combos = generate_all_combos(total_cells)

    # If combo from all_combos is valid against all equations
    # in eqs_with_cell, add to valid_combos
    valid_combos = []
    for combo in all_combos:
        combo_is_valid = True
        for equation in eqs_with_cell:
            if not verify_combo(combo, equation):
                combo_is_valid = False
                break
        if combo_is_valid:
            valid_combos.append(combo)

    # Sift through all valid_combos, discarding contradictory assignments
    # Leaves inferred free/blocked cells in
    unknown_cells = set()
    new_inferences = {}
    for combo in valid_combos:
        for cell, is_blocked in combo:
            if cell in unknown_cells:
                continue
            elif cell in new_inferences.keys() \
                    and is_blocked != new_inferences[cell]:
                del new_inferences[cell]
                unknown_cells.add(cell)
            else:
                new_inferences[cell] = is_blocked

    for cell, is_blocked in new_inferences.items():
        # Use new inferences to confirm free/blocked cells
        cell.confirmed = True
        cell.blocked = is_blocked
        # print(f"New inference: ({cell.x},{cell.y})={cell.blocked}")

        # Update sensed neighbors' B, E, H counters 
        # Attempt to basic infer on the sensed neighbors
        for x, y in alldirections:
            xx = cell.x + x
            yy = cell.y + y
            if is_in_bounds([xx, yy]) and gridworld[xx][yy].sensed:
                gridworld[xx][yy].H -= 1
                if is_blocked:
                    gridworld[xx][yy].B += 1
                else:
                    gridworld[xx][yy].E += 1
                basic_infer(gridworld[xx][yy])

        # Remove newly found inference from KB
        # Might be better but slower to recurse on all cells that appear in equations with
            # the newly confirmed cell and not just the unconfirmed neighbors
        remove_from_KB(cell)
        KB_infer_recurse_on_neighbors(cell)

def KB_infer_recurse_on_neighbors(curr):
    global alldirections
    for x, y in alldirections:
        xx = curr.x + x
        yy = curr.y + y
        if is_in_bounds([xx, yy]) and not gridworld[xx][yy].confirmed:
            KB_infer(gridworld[xx][yy])

def generate_all_combos(total_cells):
    """Generates all free/blocked combinations of a list of cells

    Args:
        total_cells (List[Cell]): list of cells to generate all combos
    Returns:
        List[Set[tuple(Cell, bool)]]: List of all combinations
    """
    
    # Generates a truth table
    truth_table_vals = list(itertools.product(
        [False, True], repeat=len(total_cells)))

    # Pairs ith value in each truth table's row with ith cell in total_cells
    all_combos = []
    for row in truth_table_vals:
        temp_set = set()
        for i in range(len(total_cells)):
            temp_set.add((total_cells[i], row[i]))
        all_combos.append(temp_set)
    return all_combos


def verify_combo(combo, equation: Equation):
    """Verifies a set of cell assignments against a known to be true equation
    Compares combo's number of blocked cells in equation vs equation's count

    Args:
        combo (set(tuple(Cell, bool))): test combination
        equation (Equation): equation from knowledge base known to be true

    Returns:
        bool: True if combination is valid, else False
    """
    num_blocked_in_eq = 0
    for cell, is_blocked in combo:
        if cell in equation.cells and is_blocked:
            num_blocked_in_eq += 1
    return num_blocked_in_eq == equation.count


def solve4test():
    return gridworld[0][0]


def get_weighted_manhattan_distance(x1, y1, x2, y2):
    """Manhattan: d((x1, y1),(x2, y2)) = abs(x1 - x2) + abs(y1 - y2)"""

    return 2*(abs(x1-x2) + abs(y1-y2))


def is_in_bounds(curr):
    """Determines whether next move is within bounds"""
    global gridworld
    return 0 <= curr[0] < len(gridworld) and 0 <= curr[1] < len(gridworld[0])


def get_num_neighbors(x, y):
    if (x == 0 and y == 0) or (x == 0 and y == dim-1) or (x == dim-1 and y == 0) or (x == dim-1 and y == dim-1):
        return 3
    elif (x == 0) or (y == 0) or (y == dim-1) or (x == dim-1):
        return 5
    else:
        return 8
