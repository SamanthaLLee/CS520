import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import solve
import time

agents = [solve.solve1]


def generate_data():

    trials_per_agent = 5
    dim = 25

    for agent_num, agent in enumerate(agents):
        print(agent_num)

        for i in range(trials_per_agent):
            # Generate and solve new gridworld with current agent
            solve.generategridworld(dim)
            test = agent()

    solve.solve1()


def generate_confusion_matrix(data, labels):
    mat = [[0 for i in range(10)] for j in range(10)]

    predictions = np.argmax(model.predict(data), axis=1)

    for i in range(data.shape[0]):
        mat[labels[i]][predictions[i]] += 1

    for i in range(10):
        print("\t".join([str(c) for c in mat[i]]))


if __name__ == "__main__":
    # generate_data()

    dim = input("What is the length of your gridworld? ")
    while not dim.isdigit() or int(dim) < 2:
        dim = input("Enter a valid length. ")

    solve.generategridworld(int(dim))
    result = solve.solve1()
    solve.printGridworld()

    if (result is None):
        print("No solution.")

    # print(solve.current_locations)
    # print(solve.input_states)
    # print(solve.output_states)

    x_train, x_test, y_train, y_test = train_test_split(
        solve.input_states, solve.output_states, test_size=0.1)

    train_in = np.reshape(x_train, (-1, 10, 10))
    test_in = np.reshape(x_test, (-1, 10, 10))
    train_out = tf.keras.utils.to_categorical(y_train, 10)
    test_out = tf.keras.utils.to_categorical(y_test, 10)

    digit_input = tf.keras.layers.Input(shape=(10, 10))
    flatten_image = tf.keras.layers.Flatten()(digit_input)
    dense_1 = tf.keras.layers.Dense(
        units=100, activation=tf.nn.relu)(flatten_image)
    dense_2 = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(dense_1)
    logits = tf.keras.layers.Dense(units=10, activation=None)(dense_2)
    probabilities = tf.keras.layers.Softmax()(logits)

    model = tf.keras.Model(inputs=digit_input, outputs=probabilities)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    generate_confusion_matrix(test_in, y_test)

    history = model.fit(train_in, train_out, epochs=20)

    generate_confusion_matrix(test_in, y_test)

    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (x_train, y_train))

    # test_dataset = tf.data.Dataset.from_tensor_slices(
    #     (x_test, y_test))

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(10, 10)),
    #     tf.keras.layers.Dense(100, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ])

    # model.compile(optimizer=tf.keras.optimizers.RMSprop(),
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True),
    #     metrics=['sparse_categorical_accuracy'])

    # model.fit(train_dataset, epochs=2)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
