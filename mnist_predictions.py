import tensorflow as tf 
import numpy as np 
import pandas as pd 

model = tf.keras.models.load_model('mnist_cnn_model.h5')

test_data = pd.read_csv('test.csv')

test_array = np.reshape(np.array(test_data), (28000, 28, 28, 1))

predictions = model.predict(test_array)

preds_df = pd.DataFrame(predictions)

def decode(row):
	for n in preds_df.columns:
		if row[n] == 1:
			return int(n)

preds_df = preds_df.apply(decode, axis=1)

preds_df = pd.DataFrame(preds_df, columns = ['Label'])
preds_df.index.names = ['ImageId']
preds_df.index += 1
preds_df.fillna(method = 'bfill', inplace = True)

print(preds_df.head())

preds_df.to_csv('mnist_submissions.csv')