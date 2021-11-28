
import random

import numpy
import pandas
import matplotlib.pyplot
import sklearn.linear_model

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
	# Headers name : Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

	# with pandas.option_context('display.max_columns', 40):
	# 	print(df.describe(include='all'))

	# Glucose, BloodPressure, SkinThickness, Insulin, BMI can't be zero!
	zero_headers = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

	for h in zero_headers :
		df[h].replace(0, df[h].median(), inplace = True)

	# with pandas.option_context('display.max_columns', 40):
	# 	print(df.describe(include='all'))

	# shufle rows
	numpy.random.seed(0)
	df.sample(frac = 1)

	# convert dataframe into numpy array
	data = df.to_numpy()

	# separate features and target values
	x = data[:, :-1]
	y = data[:, -1]

	# normalise
	x = (x - x.mean(axis = 0)) / x.std(axis = 0)

	return (x, y)

class LogisticRegression():
	
	def __init__(self, x, y, t, v) :

		assert( type(x) == numpy.ndarray and x.ndim == 2 )
		assert( type(y) == numpy.ndarray and y.ndim == 1 )
		assert( x.shape[0] == y.shape[0] )
		assert( 0 <= t <= 1 and 0 <= v <= 1 and 0 <= (t + v) <= 1 )

		self.features = x
		self.target_values = y

		# test/train ratio 
		self.train_ratio = t
		self.val_ratio = v
		self.test_ratio = 1 - (self.train_ratio + self.val_ratio)

		# test/train sets, to be calculated using self.split()
		self.train_set = None
		self.val_set = None
		self.test_set = None

		# type of gradient descent
		self.gradient_type = None

		# to store rmses for each epoch
		self.alpha = None
		self.epoch = None
		self.train_losses = None
		self.val_losses = None

		# model parameters after training
		self.fitted_parameters = None

		# to cehck if training is complete
		self.trained = False

		# to check if testing is complete
		self.tested = False

		# reults of testing
		self.test_predictions = None

		# evaluation metrics
		self.confusion_matrix = None
		self.accuracy = None
		self.precision = None
		self.recall = None
		self.f1_score = None

	def reset(self) :
		"""
		Reset all the progress made by model
		Input Parameters:
			None
		Return Values:
			None
		"""

		self.train_set = None
		self.val_set = None
		self.test_set = None
		self.gradient_type = None
		self.alpha = None
		self.epoch = None
		self.train_losses = None
		self.val_losses = None
		self.fitted_parameters = None
		self.trained = False
		self.tested = False
		self.confusion_matrix = False
		self.accuracy = None
		self.precision = None
		self.recall = None
		self.f1_score = None

	def get_splits(self) :
		"""
		Resturn the train, val and test datasets
		Input Parameters:
			None
		Return Values:
			(train_set, val_set, test_set) - a tuple containing training, validating and testing datasets
		"""

		assert(self.train_set is not None)

		return ((self.train_set[0].copy(), self.train_set[1].copy()), (self.val_set[0].copy(), self.val_set[1].copy()), (self.test_set[0].copy(), self.test_set[1].copy()))

	def split(self) :
		"""
		Splits the dataset into training, validating and testing
		Input Parameters:
			None
		Return Values:
			None
		"""

		enteries_count = self.features.shape[0]
		train_size = int( enteries_count * self.train_ratio )
		val_size = int( enteries_count * self.val_ratio )

		val_index_l = train_size
		val_index_r = train_size + val_size

		indexes = numpy.arange(enteries_count)
		# numpy.random.seed(0)
		# numpy.random.shuffle(indexes)

		# split data into train/test sets
		self.train_set = (self.features[indexes[ : train_size]], self.target_values[indexes[ : train_size]])
		self.val_set = (self.features[indexes[ val_index_l : val_index_r ]], self.target_values[indexes[ val_index_l : val_index_r ]])
		self.test_set = (self.features[indexes[ val_index_r : ]], self.target_values[indexes[ val_index_r : ]])

	def train(self, training_type, model_parameters = None, alpha = 0.1, epoch = 1000) :
		"""
		Train the model on the training set
		Input Parameters:
			training_type - type of gradient descent (BGD, SGD)
			model_parameters - initial value of model parameters
			alpha - learning rate
			epoch - number of iterations
		Return Values:
			None
		"""

		if model_parameters is None :
			model_parameters = numpy.zeros(self.features.shape[1] + 1)

		assert( training_type == "BGD" or training_type == "SGD" )
		assert( type(model_parameters) == numpy.ndarray and model_parameters.ndim == 1 and model_parameters.shape[0] == self.features.shape[1] + 1 )
		assert( type(alpha) == int or type(alpha) == float )
		assert( type(epoch) == int )

		if self.train_set is None :
			self.split()

		self.gradient_type = training_type
		self.alpha = alpha
		self.epoch = epoch

		self.train_losses = numpy.zeros(epoch)		
		self.val_losses = numpy.zeros(epoch)

		if self.gradient_type == "BGD" :
			self.batch_gradient_descent(model_parameters)
		elif self.gradient_type == "SGD" :
			self.stochastic_gradient_descent(model_parameters)

		self.trained = True
		self.tested = False

	def batch_gradient_descent(self, model_parameters) :

		# extra column to take-care for intercept coefficient
		modified_train_set_features = numpy.hstack( (numpy.ones((self.train_set[0].shape[0], 1)), self.train_set[0]) )
		modified_val_set_features = numpy.hstack( (numpy.ones((self.val_set[0].shape[0], 1)), self.val_set[0]) )

		for e in range(self.epoch) :
			train_error, train_derivative = self.cost_and_derivative(modified_train_set_features, self.train_set[1], model_parameters)
			self.train_losses[e] = train_error

			val_error, val_derivative = self.cost_and_derivative(modified_val_set_features, self.val_set[1], model_parameters)
			self.val_losses[e] = val_error

			model_parameters = model_parameters - self.alpha * train_derivative

		self.fitted_parameters = model_parameters

	def stochastic_gradient_descent(self, model_parameters) :

		# extra column to take-care for intercept coefficient
		modified_train_set_features = numpy.hstack( (numpy.ones((self.train_set[0].shape[0], 1)), self.train_set[0]) )
		modified_val_set_features = numpy.hstack( (numpy.ones((self.val_set[0].shape[0], 1)), self.val_set[0]) )

		for e in range(self.epoch) :

			random_index = random.randint(0, self.train_set[1].shape[0] - 1)

			chosen_features = numpy.array([ modified_train_set_features[ random_index ] ])
			chosen_target_values = numpy.array([ self.train_set[1][ random_index ] ])

			train_error, train_derivative = self.cost_and_derivative(chosen_features, chosen_target_values, model_parameters)
			self.train_losses[e] = train_error

			val_error, val_derivative = self.cost_and_derivative(modified_val_set_features, self.val_set[1], model_parameters)
			self.val_losses[e] = val_error

			model_parameters = model_parameters - self.alpha * train_derivative

		self.fitted_parameters = model_parameters

	def cost_and_derivative(self, x, y, m) :
		"""
		Calculates the rmse and derivative for given choice of model parameters
		Input Parameters:
			x - features
			y - target values
			m - model parameters
		Return Values:
			error - root mean square error
			derivate - value of error's derivative at current model parameters 
		"""

		n = x.shape[0]
		h = self.sigmoid(x.dot(m))
		h[ h == 0 ] = 1e-10
		h[ h ==  1 ] -= 1e-10

		error = ( -1.0 / n ) * ( numpy.sum(y * numpy.log(h) + (1 - y) * numpy.log(1.0 - h) ))
		derivative = x.T.dot(h - y)

		return (error, derivative)

	def sigmoid(self, x) :
		"""
		Calculates the sigmoid function at x
		Input Parameters:
			x - value at which to evaluate sigmoid
		Return Values:
			y - sigmoid at x
		"""

		y = 1 / ( 1 + numpy.exp(-x) )

		return y

	def plot_losses(self, save_name = None) :
		"""
		Plot the Epoch vs RMSE graph for training
		Input Parameters:
			None
		Output:
			Plot of Epoch vs RMSE
		Return Values:
			None
		"""

		assert( self.trained == True  and ( save_name is None or type(save_name) == str ))

		matplotlib.pyplot.plot(self.train_losses, label = 'Training')
		matplotlib.pyplot.plot(self.val_losses, label = 'Validation')
		matplotlib.pyplot.legend()
		matplotlib.pyplot.grid(True)
		matplotlib.pyplot.ylabel("Losses")
		matplotlib.pyplot.xlabel("Epochs")
		matplotlib.pyplot.xscale('log')
		matplotlib.pyplot.rc('axes', labelsize = 20)
		matplotlib.pyplot.title(self.gradient_type + " : " + "Epcoh vs Losses" + " at alpha = " + str(self.alpha), fontsize = 14)
		if save_name is not None :
			matplotlib.pyplot.savefig(save_name)
		matplotlib.pyplot.show()

	def test(self) :
		"""
		Calculates Error for Testing Set
		Input Parameters:
			x - features
		Return Values:
			y - target values
		"""

		assert( self.trained == True )

		self.test_predictions = self.predict(self.test_set[0])

		self.tested = True

		self.generate_metrics()


	def predict(self, x) :
		"""
		Predicts the target values for given set of features
		Input Parameters:
			x - features
		Return Values:
			y - target values
		"""

		assert( self.trained == True )
		assert( type(x) == numpy.ndarray and x.ndim == 2 and x.shape[1] + 1 == self.fitted_parameters.shape[0] )

		x_modified = numpy.hstack( (numpy.ones((x.shape[0], 1)), x) )

		y = self.sigmoid(x_modified.dot(self.fitted_parameters))
		
		# binary classification
		y[y >= 0.5] = 1
		y[y < 0.5] = 0

		return y

	def generate_metrics(self) :
		"""
		Calculates all the evaluation metrics for the model
		Input Parameters:
			None
		Return Values:
			None
		"""

		assert( self.tested == True )

		self.confusion_matrix = [[0 for i in range(2)] for j in range(2)]
		"""
		[TN][FP]
		[FN][TP]
		"""

		n = self.test_set[1].shape[0]
		for i in range(n) :
			self.confusion_matrix[int(self.test_set[1][i])][int(self.test_predictions[i])] += 1

		tn = self.confusion_matrix[0][0]
		fp = self.confusion_matrix[0][1]
		fn = self.confusion_matrix[1][0]
		tp = self.confusion_matrix[1][1]

		self.accuracy = (tp + tn) / (tn + tp + fn +fp)
		self.precision = tp / (tp + fp)
		self.recall = tp / (tp + fn)
		self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)

	def get_metrics(self) :
		"""
		Returns all the evaluation metrics for the model
		Input Parameters:
			None
		Return Values:
			(confusion_matrix, accuracy, precision, recall, f1_score)
		"""

		return (self.confusion_matrix, self.accuracy, self.precision, self.recall, self.f1_score)

def get_metrics(y_actual, y_predicted) :
	"""
	Calculates all the evaluation metrics for the model
	Input Parameters:
		y_actual - actual target values
		y_predicted - predicted target values
	Return Values:
		None
	"""

	confusion_matrix = [[0 for i in range(2)] for j in range(2)]
	"""
	[TN][FP]
	[FN][TP]
	"""

	n = test_set[1].shape[0]
	for i in range(n) :
		confusion_matrix[int(y_actual[i])][int(y_predicted[i])] += 1

	tn = confusion_matrix[0][0]
	fp = confusion_matrix[0][1]
	fn = confusion_matrix[1][0]
	tp = confusion_matrix[1][1]

	accuracy = (tp + tn) / (tn + tp + fn +fp)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2 * (precision * recall) / (precision + recall)

	return (confusion_matrix, accuracy, precision, recall, f1_score)


if __name__ == '__main__' :
	
	plots_save_location = "Plots/Q2/"

	## 1.
	dataset_name_with_path = 'Datasets/Q2/diabetes2.csv'
	
	x, y = preProccessData(dataset_name_with_path)

	# 1. a and 1. c
	model = LogisticRegression(x.copy(), y.copy(), 0.7, 0.2)
	model.split()
	train_set, val_set, test_set = model.get_splits()

	model.train("BGD")
	model.test()
	model.plot_losses(plots_save_location + "1a_BGD.jpg")
	confusion_matrix, accuracy, precision, recall, f1_score = model.get_metrics()
	print("For BDG (with default alpha):\n",confusion_matrix, accuracy, recall, f1_score)

	model.train("SGD")
	model.test()
	model.plot_losses(plots_save_location + "1a_SGD.jpg")
	confusion_matrix, accuracy, precision, recall, f1_score = model.get_metrics()
	print("For SDG (with default alpha):\n",confusion_matrix, accuracy, recall, f1_score)

	# 1. b and 1. c
	alphas = [0.01, 0.0001, 10]
	for a in alphas :
		model.train("BGD", alpha = a)
		model.test()
		model.plot_losses(plots_save_location + "_" + str(a) + "_" + "1b_BGD.jpg")
		confusion_matrix, accuracy, precision, recall, f1_score = model.get_metrics()
		print("For BDG with apha =", a, ":\n",confusion_matrix, accuracy, recall, f1_score)

		model.train("SGD", alpha = a)
		model.test()
		model.plot_losses(plots_save_location + "_" + str(a) + "_" + "1b_SGD.jpg")
		print("For SDG with apha =", a, ":\n",confusion_matrix, accuracy, recall, f1_score)

	## 2.
	model = sklearn.linear_model.SGDClassifier(random_state = 0, alpha = 0.01)
	model.fit(train_set[0], train_set[1])

	y_actual = test_set[1]
	y_predicted = model.predict(test_set[0])

	iterations_to_converge = model.n_iter_
	print("Iterations to converge for sklearn's SGDClassifier", iterations_to_converge)

	confusion_matrix, accuracy, precision, recall, f1_score = get_metrics(y_actual, y_predicted)
	print("For sklearn's SGDClassifier", confusion_matrix, precision, accuracy, recall, f1_score)
