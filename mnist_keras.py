import os

import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def preprocess():
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
    x_test = x_test.reshape(10000, 784).astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, num_classes


def train():
    batch_size = 256
    epochs = 100

    x_train, x_test, y_train, y_test, num_classes = preprocess()

    model = Sequential()
    model.add(Dense(784, activation="relu", input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(784, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=1e-4),
        metrics=["accuracy"]
    )

    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5)]
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)
    # [0.05817407166124394, 0.9824]

    json_string = model.to_json()
    open(os.path.join('./local/', 'model.json'), 'w').write(json_string)

    model.save_weights(os.path.join('./local/', 'model.hdf5'))


def predict():
    x_train, x_test, y_train, y_test, num_classes = preprocess()

    model = model_from_json(
        open(os.path.join("./local/", "model.json"), "r").read())
    model.load_weights(os.path.join("./local/", "model.hdf5"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=1e-4),
        metrics=["accuracy"]
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)


if __name__ == "__main__":
    # train()
    predict()
