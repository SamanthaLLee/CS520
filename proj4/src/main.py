import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import solve
import time
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from imblearn.under_sampling import RandomUnderSampler


agents = [solve.solve1]
in_data = []
out_data = []


def generate_data(ignore_right):

    solve.ignore_right = ignore_right

    trials_per_agent = 100
    dim = 50

    for agent_num, agent in enumerate(agents):
        # print(agent_num)

        for i in range(trials_per_agent):
            # Generate and solve new gridworld with current agent
            solve.generategridworld(dim)
            # solve.ignore_right = not solve.ignore_right
            test = agent()
    # save_to_npz()


def generate_confusion_matrix(data, labels):
    mat = [[0 for i in range(4)] for j in range(4)]

    predictions = np.argmax(model.predict(data), axis=1)

    for i in range(data.shape[0]):
        mat[labels[i]][predictions[i]] += 1

    for i in range(4):
        print("\t".join([str(c) for c in mat[i]]))


def save_to_npz():
    input = solve.input_states
    output = solve.output_states
    savez_compressed('input.npz', input)
    savez_compressed('output.npz', output)


def load_data(fromFile):
    global in_data, out_data
    if fromFile:
        loaded_in = load('input.npz')
        loaded_out = load('output.npz')
        in_data = loaded_in['arr_0']
        out_data = loaded_out['arr_0']

        # np.concatenate((in_data, solve.input_states), axis=1)
        # np.concatenate((out_data, solve.output_states))

    else:
        in_data = solve.input_states
        out_data = solve.output_states


if __name__ == "__main__":
    generate_data(False)
    load_data(False)

    # print(len(out_data))
    # print(out_data)

    # dim = input("What is the length of your gridworld? ")
    # while not dim.isdigit() or int(dim) < 2:
    #     dim = input("Enter a valid length. ")

    # solve.generategridworld(int(dim))
    # result = solve.solve1()
    # solve.printGridworld()

    # if (result is None):
    #     print("No solution.")

    weights = {0: 1.,
               1: 1.,
               2: 1.,
               3: 1.}

    # unique = [0, 1, 2, 3]

    # weights = sklearn.utils.class_weight.compute_class_weight(
    #     class_weight='balanced', classes=np.unique(unique), y=out_data)
    # weights = {i: weights[i] for i in range(4)}

    dataset = pd.DataFrame(
        {'input': in_data, 'output': out_data})
    classes_zero = dataset[dataset['output'] == 0]
    classes_one = dataset[dataset['output'] == 1]
    classes_two = dataset[dataset['output'] == 2]
    classes_three = dataset[dataset['output'] == 3]

    classes_one = classes_one.sample(len(classes_zero))
    classes_two = classes_two.sample(len(classes_zero))
    classes_three = classes_three.sample(len(classes_zero))

    print(len(classes_zero))
    print(len(classes_one))
    print(len(classes_two))
    print(len(classes_three))

    frames = [classes_one, classes_two, classes_three]

    all_dfs = pd.concat(frames)

    out_data = all_dfs.loc[:, 'output'].values
    in_data = all_dfs["input"].tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        in_data, out_data, test_size=0.1)

    train_in = np.reshape(x_train, (-1, 50, 50, 2))
    test_in = np.reshape(x_test, (-1, 50, 50, 2))
    train_out = tf.keras.utils.to_categorical(y_train, 4)
    test_out = tf.keras.utils.to_categorical(y_test, 4)

    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (train_in, train_out))
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_in, test_out))

    # BATCH_SIZE = 64
    # SHUFFLE_BUFFER_SIZE = 100

    # train_dataset = train_dataset.shuffle(
    #     SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    # test_dataset = test_dataset.batch(BATCH_SIZE)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(51, 50)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(4)
    # ])

    # model.compile(optimizer=tf.keras.optimizers.RMSprop(),
    #               loss=tf.keras.losses.CategoricalCrossentropy(
    #               from_logits=True),
    #               metrics=['categorical_accuracy'])

    # generate_confusion_matrix(test_in, y_test)
    # history = model.fit(train_dataset, epochs=20)
    # generate_confusion_matrix(test_in, y_test)

    maze_input = tf.keras.layers.Input(shape=(50, 50, 2))
    flatten_array = tf.keras.layers.Flatten()(maze_input)
    dense_1 = tf.keras.layers.Dense(
        units=100, activation=tf.nn.relu)(flatten_array)
    dense_2 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(dense_1)
    logits = tf.keras.layers.Dense(units=4, activation=None)(dense_2)
    probabilities = tf.keras.layers.Softmax()(logits)

    model = tf.keras.Model(
        inputs=maze_input, outputs=probabilities)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # GENERATING CONFUSION MATRICES

    generate_confusion_matrix(test_in, y_test)
    history = model.fit(train_in, train_out, epochs=50, class_weight=weights)
    generate_confusion_matrix(test_in, y_test)
