import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, regularizers
from keras.layers import Activation
from keras.layers import Dropout
import matplotlib.pyplot as plt

from utils import *

def get_model():
  model = Sequential()
  model.add(Dense(512))
  model.add(Activation(activation='relu', name='activation_1'))
  model.add(Dense(10, activation='softmax'))
  return model


X = np.load('new_X.npy')
y = np.load('label.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = get_model()
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

num_epochs = 70
hist = model.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), verbose=1)

y_hat = model.predict(X_test)
score = accuracy_score(y_test, y_hat)

print("Final score:", score)

train_loss = hist.history['loss']
val_loss   = hist.history['val_loss']
train_acc  = hist.history['acc']
val_acc    = hist.history['val_acc']

plot(train_loss, val_loss, "Loss changing", is_loss=True)
plot(train_acc, val_acc, "Accuracy changing", is_loss=False)
