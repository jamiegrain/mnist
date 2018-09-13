import tensorflow as tf 
import numpy as np 
import pandas as pd 

df = pd.read_csv('train.csv')

labels = df['label']
df.drop(['label'], axis = 1, inplace = True)

X = np.array(df)
y = np.array(labels)

X = tf.keras.utils.normalize(X, axis=1)

X.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, input_shape=X.shape[1:]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid))

model.compile(loss='binary_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])

model.fit(X, y, epochs = 5, validation_split = 0.2)