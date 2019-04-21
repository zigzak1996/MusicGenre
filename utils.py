import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import tensorflow as tf


def load_frozen_graph(path="net/mobilenet/mobilenet_v1_1.0_224_frozen.pb"):
    with tf.gfile.GFile(path, 'rb') as f:
        with tf.Session() as sess:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')


def split_for_vgg(file, destination_folder="npy_data"):
    X = np.load(file)
    if os.path.exists(destination_folder) == False:
        os.mkdir(destination_folder)
    for i in range(0, X.shape[0], 100):
        np.save(destination_folder + "/X" + str(i) + ".npy", X[i:i + 100])


def splitsongs(X, y, window=0.1, overlap=0.5):
    temp_X = []
    temp_y = []
    n = X.shape[0]

    chunk = int(n * window)
    offset = int(chunk * (1. - overlap))

    spsong = [X[i: min(i + chunk, n)] for i in range(0, n - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)


def to_melspectrogram(songs, n_fft=1024, hop_length=512):
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
                                                       hop_length=hop_length)[:, :, np.newaxis]

    tsongs = map(melspec, songs)

    return np.array(list(tsongs))


def read_data(src_dir, song_samples, debug=True):
    arr_specs = []
    arr_genres = []
    genres = {key: i for i, key in enumerate(sorted(os.listdir(src_dir)))}
    for genre in genres.keys():

        folder = src_dir + '/' + genre

        for file in sorted(os.listdir(folder)):
            path = folder + "/" + file
            signal, sr = librosa.load(path)
            signal = signal[:song_samples]

            if debug:
                print("Reading file: %s" % path)

            signals, y = splitsongs(signal, genres[genre])

            specs = to_melspectrogram(signals)

            arr_genres.extend(y)
            arr_specs.extend(specs)

    np.save('data.npy', np.array(arr_specs))
    np.save('label.npy', np.array(arr_genres))


def plot(train_vals, test_vals, title, is_loss=True):
    label_train = "train accuracy"
    label_test = "test accuracy"
    if is_loss:
        label_train = "train loss"
        label_test = "test loss"

    plt.plot(train_vals, '-b', label=label_train)
    plt.plot(test_vals, '-r', label=label_test)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    # plt.ylim([0, 300])
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()


def train_test_split(X, y):
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(0, 19000, 1900):
        for j in range(0, 19 * 30):
            X_test.append(X[i + j])
            y_test.append(y[i + j])
        for j in range(19 * 30, 19 * 100):
            X_train.append(X[i + j])
            y_train.append(y[i + j])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


def accuracy_score(y_test, y_hat):
    cnt = 0.
    for i in range(0, len(y_test), 19):
        vec = y_hat[i].copy()
        for j in range(1, 19):
            vec += y_hat[i + j]
        if np.argmax(vec) == y_test[i]:
            cnt += 1.
    return cnt / (len(y_test) / 19)
