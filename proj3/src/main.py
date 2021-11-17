import numpy as np
import matplotlib.pyplot as plt
import solve
import time
from terrain import Terrain

agents = [solve.solve6, solve.solve7, solve.solve8_v1, solve.solve8_v2]

actions_results = []
movements_results = []
examinations_results = []
movements_examinations_results = []
runtime_results = []
processed_results = []


def isfloat(str):
    """Determines whether a given string can be converted to float"""
    try:
        float(str)
        return True
    except ValueError:
        return False


def generate_all_graphs():
    global agents, actions_results, runtime_results, processed_results, movements_results, examinations_results, movements_examinations_results

    # Initialize constants:
    trials_per_agent = 200

    # Initialize results matrix - range[2][5] = agent 3's runtime at p=.033*5=.165
    actions_results = [0 for _ in range(len(agents))]
    movements_results = [0 for _ in range(len(agents))]
    examinations_results = [0 for _ in range(len(agents))]
    runtime_results = [0 for _ in range(len(agents))]
    processed_results = [0 for _ in range(len(agents))]
    movements_examinations_results = [0 for _ in range(len(agents))]

    # For each agent, create trials_per_p # of gridworlds
    for agent_num, agent in enumerate(agents):
        print(agent_num)
        num_fail = 0

        for i in range(trials_per_agent):
            # Generate and solve new gridworld with current agent
            solve.goal = None
            solve.start = None
            solve.actions = 0
            solve.numcellsprocessed = 0
            solve.examinations = 0
            solve.movements = 0
            solve.generategridworld(25)

            start_time = time.time()
            test = agent()
            stop_time = time.time()
            solve.printGridworld()

            if test is None:
                num_fail += 1
            else:
                runtime_results[agent_num] += stop_time - start_time
                movements_results[agent_num] += solve.movements
                examinations_results[agent_num] += solve.examinations
                actions_results[agent_num] += solve.actions
                processed_results[agent_num] += solve.numcellsprocessed
                movements_examinations_results[agent_num] += solve.movements / \
                    solve.examinations

            print(i, "done")

        # Calculate average pathlen for each agent
        num_success = trials_per_agent - num_fail
        if num_success != 0:
            actions_results[agent_num] /= num_success
            runtime_results[agent_num] /= num_success
            processed_results[agent_num] /= num_success
            movements_results[agent_num] /= num_success
            examinations_results[agent_num] /= num_success
            movements_examinations_results[agent_num] /= num_success

        print("\t" + str(num_success) +
              " gridworlds succeeded for agent = " + str(agent_num))


def plot_actions():
    global agents, actions_results

    print("actions", actions_results)

    results = actions_results
    plt.title('Agent vs. Number of Actions')
    plt.ylabel('Number of Actions')

    # ind = np.arange(2)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 6", "Agent 7"])
    # plt.xlabel('Agent Number')
    # plt.show()

    # ind = np.arange(3)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 1", "Agent 2", "Agent 3"])
    # plt.xlabel('Agent Number')
    # plt.show()

    ind = np.arange(4)
    width = .75
    plt.bar(ind, results, width, color='r')
    plt.xticks(ind, ["Agent 6", "Agent 7", "Agent 8 (v1)", "Agent 8 (v2)"])
    plt.xlabel('Agent Number')
    plt.show()


def plot_movements():
    global agents, movements_results

    print("movements", movements_results)

    results = movements_results
    plt.title('Agent vs. Number of Movements')
    plt.ylabel('Number of Movements')

    # ind = np.arange(2)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 6", "Agent 7"])
    # plt.xlabel('Agent Number')
    # plt.show()

    # ind = np.arange(3)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 1", "Agent 2", "Agent 3"])
    # plt.xlabel('Agent Number')
    # plt.show()

    ind = np.arange(4)
    width = .75
    plt.bar(ind, results, width, color='r')
    plt.xticks(ind, ["Agent 6", "Agent 7", "Agent 8 (v1)", "Agent 8 (v2)"])
    plt.xlabel('Agent Number')
    plt.show()


def plot_examinations():
    global agents, examinations_results

    print("examinations", examinations_results)

    results = examinations_results
    plt.title('Agent vs. Number of Examinations')
    plt.ylabel('Number of Examinations')

    # ind = np.arange(2)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 6", "Agent 7"])
    # plt.xlabel('Agent Number')
    # plt.show()

    # ind = np.arange(3)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 1", "Agent 2", "Agent 3"])
    # plt.xlabel('Agent Number')
    # plt.show()

    ind = np.arange(4)
    width = .75
    plt.bar(ind, results, width, color='r')
    plt.xticks(ind, ["Agent 6", "Agent 7", "Agent 8 (v1)", "Agent 8 (v2)"])
    plt.xlabel('Agent Number')
    plt.show()


def plot_movements_examinations():
    global agents, movements_examinations_results

    print("movements/examinations result", movements_examinations_results)

    results = movements_examinations_results
    plt.title('Agent vs. Movements/Examinations')
    plt.ylabel('Movements/Examinations')

    # ind = np.arange(2)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 6", "Agent 7"])
    # plt.xlabel('Agent Number')
    # plt.show()

    # ind = np.arange(3)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 1", "Agent 2", "Agent 3"])
    # plt.xlabel('Agent Number')
    # plt.show()

    ind = np.arange(4)
    width = .75
    plt.bar(ind, results, width, color='r')
    plt.xticks(ind, ["Agent 6", "Agent 7", "Agent 8 (v1)", "Agent 8 (v2)"])
    plt.xlabel('Agent Number')
    plt.show()


def plot_runtime():
    global agents, runtime_results

    print("runtime", runtime_results)

    results = runtime_results
    plt.title('Agent vs. Runtime')
    plt.ylabel('Runtime (s)')

    # ind = np.arange(2)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 6", "Agent 7"])
    # plt.xlabel('Agent Number')
    # plt.show()

    # ind = np.arange(3)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 1", "Agent 2", "Agent 3"])
    # plt.xlabel('Agent Number')
    # plt.show()

    ind = np.arange(4)
    width = .75
    plt.bar(ind, results, width, color='r')
    plt.xticks(ind, ["Agent 6", "Agent 7", "Agent 8 (v1)", "Agent 8 (v2)"])
    plt.xlabel('Agent Number')
    plt.show()


def plot_processed():
    global agents, processed_results

    print("processed", processed_results)

    results = processed_results
    plt.title('Agent vs. Number of Cells Processed')
    plt.ylabel('Number of Cells Processed')

    # ind = np.arange(2)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 6", "Agent 7"])
    # plt.xlabel('Agent Number')
    # plt.show()

    # ind = np.arange(3)
    # width = .75
    # plt.bar(ind, results, width, color='r')
    # plt.xticks(ind, ["Agent 1", "Agent 2", "Agent 3"])
    # plt.xlabel('Agent Number')
    # plt.show()

    ind = np.arange(4)
    width = .75
    plt.bar(ind, results, width, color='r')
    plt.xticks(ind, ["Agent 6", "Agent 7", "Agent 8 (v1)", "Agent 8 (v2)"])
    plt.xlabel('Agent Number')
    plt.show()


if __name__ == "__main__":

    # dim = input("What is the length of your gridworld? ")
    # while not dim.isdigit() or int(dim) < 2:
    #     dim = input("Enter a valid length. ")

    # solve.generategridworld(int(dim))
    # solve.printGridworld()
    # solve.printGridworld()
    # if (result is None):
    #     print("No solution.")

    generate_all_graphs()
    plot_actions()
    plot_movements()
    plot_examinations()
    plot_movements_examinations()
    plot_processed()
    plot_runtime()
