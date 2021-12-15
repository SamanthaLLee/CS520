import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import solve
from numpy import savez_compressed
from numpy import load
from imblearn.under_sampling import RandomUnderSampler
import math

gridworld = []
curr_state = []
actions = [0, 1, 2, 3]
model = None
location = [0, 0]


def run_p1_dense(model1):
    global model, gridworld, location
    model = model1  # tf.keras.models.load_model('./p1_cnn_model')
    gridworld = solve.generategridworld(50)
    goalreached = False
    counter = 0

    currstate = np.full((50, 50, 2), -1)

    while not goalreached:

        # planning time
        prediction_arr = model.predict(np.reshape(currstate, (-1, 50, 50, 2)))
        prediction = prediction_arr.argmax()

        currstate[location[0]][location[1]][1] = -1

        if prediction == 0:
            location[0] -= 1
            print("go up")
        elif prediction == 1:
            location[0] += 1
            print("go down")
        elif prediction == 2:
            location[1] -= 1
            print("go left")
        elif prediction == 3:
            location[1] += 1
            print("go right")

        if (location[0] == 49 and location[1] == 49):
            goalreached = True

        counter += 1

        if counter > 2000:
            break

    if not goalreached:
        print("error: inf loop")

    # get actual state by referencing gridworld


def get_actual_state(action, currstate):
    if gridworld[location[0]][location[1]].blocked:
        currstate[location[0]][location[1]][0] = 1
    else:
        currstate[location[0]][location[1]][0] = 0

    # update location
    currstate[location[0]][location[1]][1] = 1
