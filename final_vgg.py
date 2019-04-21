import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score as saccuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *

folder = "npy_output"

X = np.load(folder + "/X0.npy")

for i in range(100, 19000, 100):
    X = np.vstack((X, np.load(folder + '/X' + str(i) + ".npy")))

y = np.load("label.npy")

y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

del X, y

#NN model
input_ = tf.placeholder(shape = [None, 25088], dtype = tf.float32, name = 'input')
output = tf.placeholder(name='output', dtype = tf.int32, shape = [None])
global_step = tf.Variable(0, trainable=False)

fc1 = tf.layers.dense(inputs = input_, units = 4096, activation = tf.nn.tanh, name='layer1')
fc3 = tf.layers.dense(inputs = fc1, units = 1000, activation = tf.nn.tanh, name='layer3')
fc4 = tf.layers.dense(inputs = fc3, units = 500, activation = tf.nn.tanh, name='layer4')
fc5 = tf.layers.dense(inputs = fc4, units = 100, activation = tf.nn.tanh, name='layer5')
fc6 = tf.layers.dense(inputs = fc5, units = 50, activation = tf.nn.tanh, name= 'layer6')
fc7 = tf.layers.dense(inputs = fc6, units = 10, name='layer7_no_act')
softmax = tf.nn.softmax(logits = fc7, axis = -1)

#Optimizing step
loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = softmax, labels = output)
red_loss = tf.reduce_mean(loss_)
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.0001
# lr = 0.1
opt = tf.train.AdamOptimizer(learning_rate).minimize(red_loss)
init = tf.global_variables_initializer()

model_data = []
y_hat = None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(70):
        feed_dict = {input_: X_train, output: y_train}
        _, train_loss, acc = sess.run([opt, red_loss, softmax], feed_dict = feed_dict)
        print(f"Epoch {epoch}: loss:{train_loss} accuracy:{saccuracy_score(np.reshape(np.argmax(acc, axis = 1), newshape = [-1]), y_train)}")
        feed_dict = {input_: X_test, output: y_test}
        test_loss, acc_test = sess.run([red_loss, softmax], feed_dict = feed_dict)
        print(f"Epoch {epoch}: loss_test:{test_loss} accuracy_test:{saccuracy_score(np.reshape(np.argmax(acc_test, axis = 1), newshape = [-1]), y_test)}")
        model_data.append([train_loss, test_loss, saccuracy_score(np.reshape(np.argmax(acc, axis = 1), newshape = [-1]), y_train), saccuracy_score(np.reshape(np.argmax(acc_test, axis = 1), newshape = [-1]), y_test)])

    feed_dict = {input_: X_test, output: y_test}
    y_hat = sess.run(softmax, feed_dict=feed_dict)
del X_train, y_train
cnt = 0
y_hat = np.array(y_hat)
score = accuracy_score(y_test, y_hat)
print("Accuracy score:", score)
del X_test,y_test



model_data = np.array(model_data)
train_loss = model_data[:, 0]
val_loss   = model_data[:, 1]
train_acc  = model_data[:, 2]
val_acc    = model_data[:, 3]

plot(train_loss, val_loss, "Loss changing", is_loss=True)
plot(train_acc, val_acc, "Accuracy changing", is_loss=False)
