import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist

df = pd.read_csv('Model/Churn.csv')

X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda X: 1 if X=='Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = Sequential([
    Input(shape=(len(X_train.columns),)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='accuracy',     # or 'val_accuracy' if using validation data
    save_best_only=True,
    mode='max',
    verbose=1
)

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, batch_size=32, verbose = 0, callbacks=[checkpoint])
