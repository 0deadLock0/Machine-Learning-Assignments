#Reference: Implementing Naive Bayes- https://towardsdatascience.com/implementing-naive-bayes-algorithm-from-scratch-python-c6880cfc9c41
#Reference: ROC Curve- https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=A%20useful%20tool%20when%20predicting,values%20between%200.0%20and%201.0.

import numpy
import pandas
import matplotlib.pyplot
import sklearn.preprocessing
import sklearn.naive_bayes
import sklearn.metrics 

def preProccessData(file_name_with_relative_path):
	"""
	Preproccess the raw data for easy/better use
	Input Paramteres :
		file_name_with_relative_path - name of file from which contains data
	Return Values :
		x - features
		y - target value
	"""

	df = pandas.read_csv(file_name_with_relative_path)

	# convert dataframe into numpy array
	data = df.to_numpy()

	# cleaning data for only Trouser(1) and Pullover(2)
	data = data[ numpy.logical_or( data[: ,0] == 1, data[: ,0] == 2) ]

	# separate features and target values
	x = data[: ,1 : ]
	y = data[: ,0]

	# normalise
	x = sklearn.preprocessing.Binarizer(threshold = 127).fit_transform(x)

	return (x, y)

class NaiveBayes() :

	def __init__(self, train_set, test_set) :
		assert( type(train_set) == tuple or type(train_set) == list )
		assert( type(train_set[0]) == numpy.ndarray and train_set[0].ndim == 2 )
		assert( type(train_set[1]) == numpy.ndarray and train_set[1].ndim == 1 )
		assert( train_set[0].shape[0] == train_set[1].shape[0] )

		assert( type(test_set) == tuple or type(test_set) == list )
		assert( type(test_set[0]) == numpy.ndarray and test_set[0].ndim == 2 )
		assert( type(test_set[1]) == numpy.ndarray and test_set[1].ndim == 1 )
		assert( test_set[0].shape[0] == test_set[1].shape[0] )

		self.train_set = train_set
		self.test_set = test_set

		self.classes = None
		self.mean = None
		self.variance = None
		self.prior_probability = None
		self.fitted = False
		self.accuracy = None
	
	def fit(self):
		'''
		Fits the model on given data 
		Input Parameters:
			None
		Return Values:
			None
		'''

		self.classes = numpy.unique(self.test_set[1])
		self.mean = numpy.zeros((self.classes.shape[0], self.train_set[0].shape[1]))
		self.variance = numpy.zeros((self.classes.shape[0], self.train_set[0].shape[1]))
		self.prior_probability = numpy.zeros(self.classes.shape[0])

		for i in range(self.classes.shape[0]) :
			x = self.train_set[0][ self.train_set[1] == self.classes[i] ] #all rows having target class as classes[i] 
			self.mean[i] = x.mean(axis = 0)
			self.variance[i] = x.var(axis = 0)
			self.prior_probability[i] = x.shape[0] / self.train_set[0].shape[0]

		self.mean[ self.mean == 0 ] = 1e-10
		self.variance[ self.variance == 0 ] = 1e-10
		self.prior_probability[ self.prior_probability == 0 ] = 1e-10

		self.fitted = True

	def gaussian(self, x, mean, variance) :
		"""
		Calculates Gaussian at given inputs
		Input Parameters:
			x - input for which to calculate Gaussian
			mean - mean of distribution
			variance - variance of distribution
		Return Values:
			gaussian_value - gaussian at x, under given mean and variance
		"""

		numerator = numpy.exp( -( x - mean ) ** 2 / ( 2 * variance ))
		denominator = numpy.sqrt(2 * numpy.pi * variance)

		gaussian_value = numerator / denominator
		return gaussian_value

	def calculate_likelihood(self, class_i, x) :
		"""
		Calculates the likelihood of x being mapped to class_i
		Input Parameters:
			x - input for which to check its class
			class_i - class against which to check x
		Return Values:
			likelihood - expected likelihood of x being mapped to class_i
		"""

		class_mean = self.mean[class_i]
		class_variance = self.variance[class_i]
		likelihood = self.gaussian(x, class_mean, class_variance)

		return likelihood

	def predict_class(self, x):
		"""
		Predicts the class for given input (single)
		Input Parameters:
			None
		Return Values:
			predicted_class - Predicted class for input
		"""

		posteriors = numpy.zeros(self.classes.shape[0])
		for i in range(self.classes.shape[0]) :
			prior = numpy.log( self.prior_probability[i] )
			conditional = numpy.sum( numpy.log( self.calculate_likelihood(i, x) ) )
			posterior = prior + conditional
			posteriors[i] = posterior

		predicted_class = self.classes[numpy.argmax(posteriors)]

		return predicted_class

	def test(self) :
		"""
		Tests the model on test_set
		Input Parameters:
			None
		Return Values:
			None
		"""

		assert( self.fitted == True )

		predictions = self.predict(self.test_set[0])

		self.calculate_accuracy(self.test_set[1], predictions)

	def predict(self, x):
		"""
		Make predictions for given input
		Input Parameters:
			x - given input
		Return Values:
			y_pred - Predicted Classes
		"""

		y_pred = numpy.zeros(x.shape[0])
		for i in range(x.shape[0]) :
			y_pred[i] = self.predict_class(x[i])
		
		return y_pred

	def calculate_accuracy(self, y_actual, y_prediction) :
		"""
		Calculates the accuracy of given predictions.
		Input Parameters:
			y_actual - actual classes
			y_prediction - predicted clases
		Return Value:
			None
		"""

		self.accuracy = numpy.count_nonzero(y_actual == y_prediction) / y_actual.shape[0] * 100

	def get_accuracy(self) :
		"""
		Calculates the accuracy of given predictions.
		Input Parameters:
			None
		Return Values:
			None
		"""

		assert( self.accuracy is not None )

		return self.accuracy

def print_metrics(y_testing, y_predicted) :
	"""
	Prints the metric for sklearn Naive Bayes model
	Input Parameters:
		y_testing- orginal classes
		y_predicted- predicted classes
	Return Values:
		None
	"""

	accur = sklearn.metrics.accuracy_score(y_testing, y_predicted)
	precision = sklearn.metrics.precision_score(y_testing, y_predicted)
	recall = sklearn.metrics.recall_score(y_testing, y_predicted)
	conf_matrix = sklearn.metrics.confusion_matrix(y_testing, y_predicted)
	
	print("Confusion Matrix for sklearn's Naive Bayes", conf_matrix)
	print("Accuracy for sklearn's Naive Bayes", accur)
	print("Precision for sklearn's Naive Bayes", precision)
	print("Recall for sklearn's Naive Bayes", recall)
	

def plot_roc_curve(model, test_set, y_predicted) :
	"""
	Plots ROC for sklearn Naive Bayes model
	Input Parameters:
		model - model which is used
		test_set- test set used for testing
		y_predicted- predicted classes
	Return Values:
		None
	"""

	nb_probs = model.predict_proba(test_set[0])
	# keep probabilities for the positive outcome only
	nb_probs = nb_probs[:, 1]
	# calculate roc curves
	nb_fpr, nb_tpr, _ = sklearn.metrics.roc_curve(test_set[1], nb_probs, pos_label=2)
	
	matplotlib.pyplot.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes')
	matplotlib.pyplot.title("ROC Curve")
	matplotlib.pyplot.xlabel('False Positive Rate')
	matplotlib.pyplot.ylabel('True Positive Rate')
	matplotlib.pyplot.legend()
	matplotlib.pyplot.savefig("Plots/Q3/ROC_Cruve.jpg")
	matplotlib.pyplot.show()

if __name__ == "__main__" :

	train_dataset_name_with_path = 'Datasets/Q3/fashion-mnist_train.csv'
	test_dataset_name_with_path = 'Datasets/Q3/fashion-mnist_test.csv'

	train_set = preProccessData(train_dataset_name_with_path)
	test_set = preProccessData(test_dataset_name_with_path)

	## 3. 1
	model = NaiveBayes((train_set[0].copy(), train_set[1].copy()), (test_set[0].copy(), test_set[1].copy()))
	model.fit()
	model.test()
	print("Accuracy of model is", model.get_accuracy())

	## 3. 2
	new_set = ( numpy.vstack( (train_set[0], test_set[0]) ),  numpy.concatenate( [train_set[1], test_set[1]] ) )  
	num_folds = 5
	subset_size = new_set[0].shape[0] / num_folds

	accuracySum = 0
	for i in range(num_folds) :

		l = int(i * subset_size)
		r = int((i + 1) * subset_size)

		testing_set = ( new_set[0][ l : r ], new_set[1][ l : r ])
		training_set = ( numpy.vstack( (new_set[0][ : l], new_set[0][ r : ]) ),  numpy.concatenate( [new_set[1][ : l], new_set[1][ r : ] ] ) ) 

		model2 = NaiveBayes((training_set[0].copy(), training_set[1].copy()), (testing_set[0].copy(), testing_set[1].copy()))
		model2.fit()
		model2.test()
		accuracySum += model2.get_accuracy()

	print("Accuracy of k-fold validation is", accuracySum / num_folds)	

	## 3. 3
	model3 = sklearn.naive_bayes.GaussianNB()
	model3.fit(train_set[0].copy(), train_set[1].copy())
	y_predicted = model3.predict(test_set[0].copy())

	print_metrics(test_set[1], y_predicted) #Confusion Matrix and Accuracy, Precision, Recall
	plot_roc_curve(model3, test_set, y_predicted) # ROC curve
