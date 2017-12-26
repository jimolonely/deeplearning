import tensorflow as tf
import numpy as np

TRAIN_DATA_FILE = "train_data.csv"
TEST_DATA_FILE = "test_data.csv"


def read_split_train_test_data():
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=TRAIN_DATA_FILE,
        target_dtype=np.int,
        features_dtype=np.float32,
        target_column=0
    )
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=TEST_DATA_FILE,
        target_dtype=np.int,
        features_dtype=np.float32,
        target_column=0
    )
    # print(training_set)
    X, y = training_set.data, training_set.target
    print(X.shape)
    print(y.shape)
    X_test, y_test = test_set.data, test_set.target
    print(X_test.shape)
    print(y_test.shape)
    return X, y, X_test, y_test


def read_train_test_data():
    '''不把标签和数据分开'''
    x, y, x_t, y_t = read_split_train_test_data()
    y = y.reshape(x.shape[0], -1)
    print(y.shape)
    y_t = y_t.reshape(x_t.shape[0], -1)
    x_n = np.concatenate((x, y), axis=1)
    y_n = np.concatenate((x_t, y_t), axis=1)
    print(x_n.shape)
    return x_n, y_n
    # return 1,2


def split_data_label(X):
    '''将数据和标签分开,默认标签位于最后一列'''
    data, label = X[:, :-1], X[:, -1]
    return data, label
