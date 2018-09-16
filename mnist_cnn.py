import tensorflow as tf 
import numpy as np 

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

X.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential()
<<<<<<< HEAD
model.add(tf.keras.layers.Conv2D(100, (3,3), activation = tf.nn.relu, input_shape=(28, 28, 1)))
=======
model.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, input_shape=X.shape[1:]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu))
>>>>>>> fed121d8d02547479056aa9dea95844968731645
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())

<<<<<<< HEAD
model.add(tf.keras.layers.Dense(16, activation = tf.nn.relu))
=======
model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))
>>>>>>> fed121d8d02547479056aa9dea95844968731645

model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(loss='sparse_categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 2, validation_data = (x_test, y_test))

model.save('mnist_cnn_model.h5')