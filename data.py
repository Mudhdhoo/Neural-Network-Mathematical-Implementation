import tensorflow as tf
from keras.datasets import mnist
from tensorflow.python.ops.numpy_ops import np_config 
np_config.enable_numpy_behavior()

# Data preprocessing

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.reshape(x_train, [60000, 1, 784])
x_train = x_train / 255
y_train = tf.keras.utils.to_categorical(y_train)

x_test = tf.reshape(x_test, [x_test.shape[0], 1, 784])
x_test = x_test / 255
y_test = tf.keras.utils.to_categorical(y_test)
