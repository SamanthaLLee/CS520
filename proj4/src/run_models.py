import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import solve
from numpy import savez_compressed
from numpy import load
from imblearn.under_sampling import RandomUnderSampler
import copy
import time

gridworld = []
curr_state = []
actions = [0, 1, 2, 3]
model = None
location = [0, 0]
opp_directions = {1: 0, 0: 1, 2: 3, 3: 2}
cardinaldirections = [(1, 0), (-1, 0), (0, 1), (0, -1)]
totalplanningtime = 0
trajectorylen = 0
numplans = 0

trajectory_results = []
runtime_results = []
planning_results = []


def compare_agents(modelname):
    global model
    model = tf.keras.models.load_model(modelname)

    if "p1" in modelname:
        generate_all_data(1)
    elif "p2" in modelname:
        generate_all_data(2)


def run_model():
    global model, gridworld, location, trajectorylen, totalplanningtime, numplans

    location = [0, 0]

    gridworld = solve.gridworld
    goalreached = False
    counter = 0
    prediction = -1

    actions_tried = {}

    currstate = np.full((50, 50, 2), -1)
    currstate[0][0][0] = 1

    while not goalreached:

        # print(location)
        trajectorylen += 1

        # Plan the next step

        start = time.time()
        prediction_arr = model.predict(np.reshape(currstate, (-1, 50, 50, 2)))
        ind = prediction_arr.argsort()[-4:][::-1]
        i = len(ind[0]) - 1
        prediction = ind[0][i]
        end = time.time()

        totalplanningtime += end-start
        numplans += 1

        while True:
            # Save starting location
            initial_location = copy.deepcopy(location)

            # Indicate that we are leaving current location
            currstate[location[0]][location[1]][1] = -1

            # Update location
            if prediction == 0:
                location[0] -= 1
                # print("go up")
            elif prediction == 1:
                location[0] += 1
                # print("go down")
            elif prediction == 2:
                location[1] -= 1
                # print("go left")
            elif prediction == 3:
                location[1] += 1
                # print("go right")

            # Check if the update is valid
            if not is_in_bounds(location[0], location[1]):
                # If out of bounds, undo update and get next prediction
                location = initial_location

                i -= 1
                if i < 0:
                    print("No more valid predictions")
                    return

                prediction = ind[0][i]
                # print("new prediction", prediction)
                # print("location", location)
            elif tuple(initial_location) in actions_tried and prediction in actions_tried[tuple(initial_location)]:
                location = initial_location
                i -= 1
                if i < 0:
                    break
                prediction = ind[0][i]
            else:
                if tuple(initial_location) in actions_tried:
                    actions_tried[tuple(initial_location)].append(prediction)
                else:
                    actions_tried[tuple(initial_location)] = [prediction]
                break

        if (location[0] == 49 and location[1] == 49):
            goalreached = True

        # Query true gridworld to get current state
        currstate = get_actual_state(currstate)

        counter += 1
        if counter > 4000:
            break

    if not goalreached:
        print("error: inf loop")


def get_actual_state(currstate):
    global model, gridworld, location

    if gridworld[location[0]][location[1]].blocked:
        currstate[location[0]][location[1]][0] = 1
    else:
        currstate[location[0]][location[1]][0] = 0

    # Mark blocked neighbors as seen
    for dx, dy in cardinaldirections:
        xx, yy = location[0] + dx, location[1] + dy
        if is_in_bounds(xx, yy) and gridworld[xx][yy].blocked:
            currstate[xx][yy][0] = 1

    # update location
    currstate[location[0]][location[1]][1] = 1

    return currstate


def is_in_bounds(x, y):
    """Determines whether next move is within bounds"""
    global gridworld
    return 0 <= x < 50 and 0 <= y < 50


def generate_all_data(projnum):
    global trajectory_results, trajectorylen, totalplanningtime, planning_results, runtime_results, numplans

    # Initialize constants:
    trials_per_agent = 100

    # Initialize results matrix
    trajectory_results = [
        [0 for _ in range(2)] for _ in range(trials_per_agent)]
    runtime_results = [[0 for _ in range(2)] for _ in range(trials_per_agent)]
    planning_results = [[0 for _ in range(2)] for _ in range(trials_per_agent)]

    # For each agent, create trials_per_p # of gridworlds

    for i in range(trials_per_agent):
        # Generate and solve new gridworld with current agent
        solve.generategridworld(50)

        solve.trajectorylen = 0
        solve.totalplanningtime = 0
        trajectorylen = 0
        totalplanningtime = 0

        start_time_og = time.time()
        if projnum == 1:
            solve.solve1()
        else:
            solve.solve2()
        stop_time_og = time.time()

        start_time_ml = time.time()
        run_model()
        stop_time_ml = time.time()

        trajectory_results[i][0] = solve.trajectorylen
        runtime_results[i][0] = stop_time_og - start_time_og
        planning_results[i][0] = solve.totalplanningtime / solve.numplans

        trajectory_results[i][1] = trajectorylen
        runtime_results[i][1] = stop_time_ml - start_time_ml
        planning_results[i][1] = totalplanningtime / numplans

        print(i, "done")

    print("\t" "agents done")

    print(trajectory_results)
    print(runtime_results)
    print(planning_results)

    plot_traj()
    plot_runtime()
    plot_planning()


def plot_traj():
    global trajectory_results

    plt.title('Trial vs. Trajectory')
    plt.xlabel('Trial Number')
    plt.ylabel('Trajectory Length')

    plt.scatter(np.arange(100), trajectory_results[0])
    plt.scatter(np.arange(100), trajectory_results[1])
    plt.legend(["Original Agent", "ML Agent"])

    plt.show()


def plot_runtime():
    global runtime_results
    plt.title('Trial vs. Runtime')
    plt.xlabel('Trial Number')
    plt.ylabel('Runtime (s)')
    # plotting the column as histogram
    plt.scatter(np.arange(100), runtime_results[0])
    plt.scatter(np.arange(100), runtime_results[1])
    plt.legend(["Original Agent", "ML Agent"])
    plt.show()


def plot_planning():
    global planning_results
    plt.title('Trial vs. Average Time to Plan')
    plt.xlabel('Trial Number')
    plt.ylabel('Average Time Per Planning Step (s)')
    # plotting the column as histogram
    plt.scatter(np.arange(100), planning_results[0])
    plt.scatter(np.arange(100), planning_results[1])
    plt.legend(["Original Agent", "ML Agent"])
    plt.show()
