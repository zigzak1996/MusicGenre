import numpy as np

from keras import models, layers
from sklearn.preprocessing import LabelEncoder
from utils import *

def create_model():
    model = models.Sequential()

    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], )
    return model

X = np.load("new_X.npy")
y = np.load("label.npy")
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = create_model()
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
