from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import models, Input
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir='logs/mnist_run', histogram_freq=1)

model.fit(X_train, y_train, epochs=5, validation_split=0.1, verbose=0, callbacks=[tensorboard_callback])
