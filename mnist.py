import numpy as np 
import pandas as pd 
import tensorflow as tf 

df = pd.read_csv('train.csv')

labels = df['label']
df.drop(['label'], axis = 1, inplace = True)

X = np.array(df)
y = np.array(labels)

X = tf.keras.utils.normalize(X, axis=1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu, input_dim = (784,)))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(loss = 'sparse_categorical_crossentropy',
	optimizer = 'adam',
	metrics = ['accuracy'])

model.fit(X, y, validation_split = 0.2, epochs=4)

df_test = pd.read_csv('test.csv')

X_sub = tf.keras.utils.normalize(np.array(df_test), axis=1)

y_sub = model.predict(X_sub)
y_sub = np.argmax(y_sub, axis = 1)

predictions = pd.DataFrame(y_sub, index = df_test.index)
predictions.index += 1
predictions.rename(columns = {0: 'Label'}, inplace = True)
predictions.index.names = ['ImageId']

print(predictions.head())

predictions.to_csv('mnist_submission.csv')
sub_check = pd.read_csv('mnist_submission.csv', index_col = 0)
print(sub_check.head())