import numpy as np
from keras.datasets import mnist

class DataUtils:	

	# Initialization
	def __init__(self):
		pass

	# Turn a multi-class label vector into anomaly/normal binary label vector
	@staticmethod
	def anomaly(y, anomaly_label):
		'''
			input: [1, 2, 3, 4, 5, 6, 5, 7]
			anomaly_label = 2
			output: [0, 1, 0, 0, 0, 0, 0, 0]
		'''
		abnomaly_index = np.where(y==anomaly_label)
		normal_index = np.where(y!=anomaly_label)
		y[abnomaly_index] = 1
		y[normal_index] = 0
		return y

if __name__ == '__main__': 
	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	print DataUtils.anomaly(y_train, 5)
