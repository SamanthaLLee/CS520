import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import solve
import time

agents = [solve.solve1]


def generate_data():

    trials_per_agent = 20
    dim = 25

    for agent_num, agent in enumerate(agents):
        print(agent_num)

        for i in range(trials_per_agent):
            # Generate and solve new gridworld with current agent
            solve.generategridworld(dim)
            test = agent()


def generate_confusion_matrix(data, labels):
    mat = [[0 for i in range(4)] for j in range(4)]

    predictions = np.argmax(model.predict(data), axis=1)

    for i in range(data.shape[0]):
        mat[labels[i]][predictions[i]] += 1

    for i in range(4):
        print("\t".join([str(c) for c in mat[i]]))


if __name__ == "__main__":
    generate_data()

    # dim = input("What is the length of your gridworld? ")
    # while not dim.isdigit() or int(dim) < 2:
    #     dim = input("Enter a valid length. ")

    # solve.generategridworld(int(dim))
    # result = solve.solve1()
    # solve.printGridworld()

    # if (result is None):
    #     print("No solution.")

    x_train, x_test, y_train, y_test = train_test_split(
        solve.input_states, solve.output_states, test_size=0.1)

    train_in = np.reshape(x_train, (-1, 25, 25, 4))
    test_in = np.reshape(x_test, (-1, 25, 25, 4))
    train_out = tf.keras.utils.to_categorical(y_train, 4)
    test_out = tf.keras.utils.to_categorical(y_test, 4)

    maze_input = tf.keras.layers.Input(shape=(25, 25, 4))
    flatten_array = tf.keras.layers.Flatten()(maze_input)
    dense_1 = tf.keras.layers.Dense(
        units=100, activation=tf.nn.relu)(flatten_array)
    dense_2 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(dense_1)
    logits = tf.keras.layers.Dense(units=4, activation=None)(dense_2)
    probabilities = tf.keras.layers.Softmax()(logits)

    model = tf.keras.Model(inputs=maze_input, outputs=probabilities)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # GENERATING CONFUSION MATRICES

    generate_confusion_matrix(test_in, y_test)
    history = model.fit(train_in, train_out, epochs=60)
    generate_confusion_matrix(test_in, y_test)
