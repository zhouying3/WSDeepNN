import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from keras.datasets import mnist
from keras.utils import np_utils
#from keras.layers import Input, Dense, Dropout
#from keras.models import Model, load_model
#from keras.callbacks import ModelCheckpoint
from DataUtils import DataUtils

class MLModel:
	
	# Initialization
	def __init__(self, nb_classes, 
			seed=False):
		if seed:		
			np.random.seed(0)
		self.nb_classes = nb_classes
		self.model = None

	# Configuration
	def config(self, clf="dt"):
		if clf.lower() == "dt":
			self.model = tree.DecisionTreeClassifier()
		elif clf.lower() == "svm":
			self.model = svm.SVC(kernel='rbf',
					probability=True)
		else:
			self.model = tree.DecisionTreeClassifier()

	# Fit
	def fit(self, X_train, y_train):
		self.model.fit(X_train, y_train)

	def _proba(self, X_test):
		return self.model.predict_proba(X_test)

	def _predict(self, X_test):
		return self.model.predict(X_test)

	def predict_proba(self, X_test):
		return self._proba(X_test)

	def predict(self, X_test):
		return self._predict(X_test)

	def evaluate(self, X_test, y_test):
		Y_test = np_utils.to_categorical(y_test, self.nb_classes)
		y_pred = self.predict(X_test)
		y_proba = self.predict_proba(X_test)
		confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
		precision = metrics.precision_score(y_test, y_pred)
		recall = metrics.recall_score(y_test, y_pred)
		#specificity = specificity_score(y_test, y_pred)
		#gmean = np.sqrt(recall * specificity)
		f1 = metrics.f1_score(y_test, y_pred)
		pr_auc = metrics.average_precision_score(Y_test, y_proba)
		roc_auc = metrics.roc_auc_score(Y_test, y_proba)		
		return confusion_matrix, precision, recall, f1, pr_auc, roc_auc
		
if __name__ == '__main__': 

	# this function turns the label vector into anomaly vector
	def anomaly(y_train, y_test, anomaly_label):
		y_train = DataUtils.anomaly(y_train, anomaly_label)
		y_test = DataUtils.anomaly(y_test, anomaly_label)
		return y_train, y_test

	# this function prints the metrics in CSV format
	def show(score):
		confusion_matrix, precision, recall, f1, prc_auc, roc_auc = score	
		print "TN,FP,FN,TP,Precision,Recall,F1,PRC,ROC"
 		print "%d,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f" \
			%(confusion_matrix[0,0], confusion_matrix[0,1], 
			confusion_matrix[1,0], confusion_matrix[1,1],
			precision, recall, f1, prc_auc, roc_auc)

		
	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	
	# set the anomaly label
	anomaly_label = 9
	# modify the y_train and y_test
	y_train, y_test = anomaly(y_train, y_test, anomaly_label)
	
	# preprocess
	input_dim = 784
	X_train = X_train.reshape(60000, input_dim)
	X_test = X_test.reshape(10000, input_dim)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')
	
	# normalization
	X_train /= 255
	X_test /= 255
		
	# obtain the number of classes	
	nb_classes = np.size(np.unique(y_train))
	print "Anomaly label: %d" %(anomaly_label)
	print "Number of classes: %d" %(nb_classes)

	# training
	mlModel = MLModel(nb_classes, seed=True)
	mlModel.config(clf='svm')
	mlModel.fit(X_train, y_train)

	# evaluate on training data
	print "Training"
	score = mlModel.evaluate(X_train, y_train)	
	show(score)	

	# evaluate on testing data
	print "Testing"
	score = mlModel.evaluate(X_test, y_test)
	show(score)	
	
