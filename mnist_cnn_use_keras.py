import os

import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
from keras import backend as K


def preprocess():
    num_classes = 10
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, input_shape, num_classes


def train():
    batch_size = 128
    epochs = 100

    x_train, x_test, y_train, y_test, input_shape, num_classes = preprocess()

    model = Sequential()
    model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=input_shape
    ))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adadelta(),
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5)]
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)
    # [0.028514785393566125, 0.9917]

    json_string = model.to_json()
    open(os.path.join('./local/', 'model_cnn.json'), 'w').write(json_string)

    model.save_weights(os.path.join('./local/', 'model_cnn.hdf5'))


def predict():
    x_train, x_test, y_train, y_test, input_shape, num_classes = preprocess()

    model = model_from_json(
        open(os.path.join("./local/", "model_cnn.json"), "r").read()
    )
    model.load_weights(os.path.join("./local/", "model_cnn.hdf5"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adadelta(),
        metrics=["accuracy"]
    )

    print(model.evaluate(x_test, y_test))


if __name__ == "__main__":
    # train()
    predict()
