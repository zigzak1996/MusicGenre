import sys
import cv2
from utils import *

read_data(sys.argv[1], 660000)

load_frozen_graph()

graph_input = tf.get_default_graph().get_tensor_by_name("input:0")
graph_output = tf.get_default_graph().get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')

X = np.load("data.npy")
j = 0
result = []
data = []
while j < X.shape[0]:
    data.append(cv2.resize(X[j], (224, 224)))
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

np.save("new_X.npy", np.array(result))
