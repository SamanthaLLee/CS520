import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import solve
from numpy import savez_compressed
from numpy import load
import run_models

in_data = []
out_data = []
model = None


def generate_data(agentnum):
    trials_per_agent = 300
    dim = 50
    for i in range(trials_per_agent):
        # Generate and solve new gridworld with current agent
        solve.generategridworld(dim)
        if agentnum == 1:
            solve.solve1()
        else:
            solve.solve2()

    print(len(solve.input_states))
    print(len(solve.output_states))
    # save_to_npz()


def generate_confusion_matrix(data, labels):
    mat = [[0 for i in range(4)] for j in range(4)]

    predictions = np.argmax(model.predict(data), axis=1)

    for i in range(data.shape[0]):
        mat[labels[i]][predictions[i]] += 1

    for i in range(4):
        print("\t".join([str(c) for c in mat[i]]))


def save_to_npz():
    # not currently using
    input = solve.input_states
    output = solve.output_states
    savez_compressed('input.npz', input)
    savez_compressed('output.npz', output)


def load_data(fromFile):
    # not currently using
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


def undersample_data():
    global in_data, out_data

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


def p1_dense():

    global in_data, out_data, model
    opt = input("Are you creating a new model? Y/N ")

    createModel = False
    if opt == 'Y' or opt == 'y':
        createModel = True

    # loadDataFromFile = False
    # load_data(loadDataFromFile)

    generate_data(1)
    in_data = solve.input_states
    out_data = solve.output_states
    undersample_data()

    # print(len(solve.input_states))
    # print(len(solve.output_states))

    filename = 'p1_dense_history_log.csv'
    history_logger = tf.keras.callbacks.CSVLogger(
        filename, separator=",", append=True)

    x_train, x_test, y_train, y_test = train_test_split(
        in_data, out_data, test_size=0.5)

    # Solve1
    train_in = np.reshape(x_train, (-1, 50, 50, 2))
    test_in = np.reshape(x_test, (-1, 50, 50, 2))

    train_out = tf.keras.utils.to_categorical(y_train, 4)
    test_out = tf.keras.utils.to_categorical(y_test, 4)

    if createModel:
        maze_input = tf.keras.layers.Input(shape=(50, 50, 2))
        flatten_array = tf.keras.layers.Flatten()(maze_input)
        dense_1 = tf.keras.layers.Dense(
            units=100, activation=tf.nn.relu)(flatten_array)
        dense_2 = tf.keras.layers.Dense(
            units=50, activation=tf.nn.relu)(dense_1)
        logits = tf.keras.layers.Dense(units=4, activation=None)(dense_2)
        probabilities = tf.keras.layers.Softmax()(logits)

        model = tf.keras.Model(
            inputs=maze_input, outputs=probabilities)

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model = tf.keras.models.load_model('./p1_dense_model')
        # model.compile(optimizer='adam', loss='categorical_crossentropy',
        #               metrics=['categorical_accuracy'])

    # GENERATING CONFUSION MATRICES

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

    generate_confusion_matrix(test_in, y_test)
    history = model.fit(train_in, train_out, validation_split=0.33,
                        epochs=500, batch_size=64,
                        callbacks=[callback, history_logger],
                        workers=0,
                        shuffle=True)
    generate_confusion_matrix(test_in, y_test)

    results = model.evaluate(test_in, test_out, batch_size=128)
    print("test loss, test acc:", results)

    predictions = model.predict(test_in[:3])
    print("predictions shape:", predictions.shape)

    # model.save("./p1_dense_model2")

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def p1_cnn():
    global in_data, out_data, model

    opt = input("Are you creating a new model? Y/N ")

    createModel = False
    if opt == 'Y' or opt == 'y':
        createModel = True

    generate_data(1)
    in_data = solve.input_states
    out_data = solve.output_states
    undersample_data()

    filename = 'p1_cnn_history_log.csv'
    history_logger = tf.keras.callbacks.CSVLogger(
        filename, separator=",", append=True)

    x_train, x_test, y_train, y_test = train_test_split(
        in_data, out_data, test_size=0.5)

    train_in = np.reshape(x_train, (-1, 50, 50, 2))
    test_in = np.reshape(x_test, (-1, 50, 50, 2))

    train_out = tf.keras.utils.to_categorical(y_train, 4)
    test_out = tf.keras.utils.to_categorical(y_test, 4)

    if createModel:
        maze_input = tf.keras.layers.Input(shape=(50, 50, 2))
        cnn_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                       padding="valid", activation=tf.nn.relu)(maze_input)
        flatten_array = tf.keras.layers.Flatten()(cnn_1)
        dropout_1 = tf.keras.layers.Dropout(rate=0.5)(flatten_array)
        dense_1 = tf.keras.layers.Dense(
            units=50, activation=tf.nn.relu)(dropout_1)
        logits = tf.keras.layers.Dense(units=4, activation=None)(dense_1)
        probabilities = tf.keras.layers.Softmax()(logits)

        model = tf.keras.Model(
            inputs=maze_input, outputs=probabilities)

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model = tf.keras.models.load_model('./p1_cnn_model')

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    generate_confusion_matrix(test_in, y_test)
    history = model.fit(train_in, train_out, validation_split=0.33,
                        epochs=30, batch_size=64,
                        callbacks=[callback, history_logger],
                        workers=0,
                        shuffle=True)
    generate_confusion_matrix(test_in, y_test)

    run_models.run_p1_dense(model)

    results = model.evaluate(test_in, test_out, batch_size=128)
    print("test loss, test acc:", results)

    predictions = model.predict(test_in[:3])
    print("predictions shape:", predictions.shape)

    model.save("./p1_cnn_model")

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def p2_dense():
    print("hi")


def p2_cnn():
    print("hi")


if __name__ == "__main__":

    # run_models.run_p1_dense()

    print("1: Project 1 - Full Dense Layers")
    print("2: Project 1 - Convolutional Neural Network")
    print("3: Project 2 - Full Dense Layers")
    print("4: Project 2 - Convolutional Neural Network")
    opt = input("What NN would you like to train? ")
    while not opt.isdigit() or int(opt) < 1 or int(opt) > 4:
        opt = input("Enter a valid option. ")

    if opt == '1':
        p1_dense()
    elif opt == '2':
        p1_cnn()
    elif opt == '3':
        p2_dense()
    elif opt == '4':
        p2_cnn()
    else:
        print("Error.")
