import cv2
import sys
from utils import *

read_data(sys.argv[1], 660000)

load_frozen_graph('net/inception/InceptionV3.pb')

graph_input = tf.get_default_graph().get_tensor_by_name("Model_Input:0")
graph_output = tf.get_default_graph().get_tensor_by_name('Model_Output:0')

X = np.load("data.npy")

data = []
result = []
j = 0
while j < X.shape[0]:
    data.append(cv2.resize(X[j], (299, 299)))
    if len(data) == 25:
        data = np.array(data)
        data = np.squeeze(np.stack((data,) * 3, -1))
        with tf.Session() as sess:
            new_input = sess.run(graph_output, feed_dict={graph_input: data})
            for i in new_input:
                result.append(i)
            del new_input
        data = []
    j += 1
    if j % 100  == 0:
        print(j)

if len(data) > 0:
    data = np.array(data)
    data = np.squeeze(np.stack((data,) * 3, -1))
    with tf.Session() as sess:
        new_input = sess.run(graph_output, feed_dict={graph_input: data})
        for i in new_input:
            result.append(i)
        del new_input
result = np.array(result)

np.save('new_X.npy', result)
