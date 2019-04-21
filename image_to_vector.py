import cv2
import sys

from utils import *

load_frozen_graph('net/vgg/vgg16.tfmodel')

graph_input = tf.get_default_graph().get_tensor_by_name("images:0")
graph_output = tf.get_default_graph().get_tensor_by_name('fc6/Reshape:0')

X = np.load('npy_data/' + sys.argv[1])

data = []
result = []
j = 0

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
    if j % 25  == 0:
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
if os.path.exists("npy_output") == False:
	os.mkdir("npy_output")
    
np.save('npy_output/' + sys.argv[1], result)
