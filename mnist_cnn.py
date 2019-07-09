import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def train():
    batch_size = 256
    num_classes = 10
    epochs = 100

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
    x_test = x_test.reshape(10000, 784).astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

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

    json_string = model.to_json()
    open(os.path.join('./local/', 'model.json'), 'w').write(json_string)

    model.save_weights(os.path.join('./local/', 'model.hdf5'))


if __name__ == "__main__":
    # download()
    train()
