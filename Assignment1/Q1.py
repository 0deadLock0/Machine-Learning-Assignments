
import numpy
import pandas
import matplotlib.pyplot
import sklearn.linear_model
import sklearn.model_selection

def preProccessData(file_name_with_relative_path):
	"""
	Preproccess the raw data for easy/better use
	Input Paramteres :
		file_name_with_relative_path - name of file from which contains data
	Return Values :
		x - features
		y - target value
	"""

	df = pandas.read_csv(file_name_with_relative_path, header = None)

	# convert Sex values into numerical
	df[0].replace('M', 1, inplace = True) 
	df[0].replace('F', 2, inplace = True)
	df[0].replace('I', 3, inplace = True)

	# shufle rows
	numpy.random.seed(0)
	df.sample(frac = 1)

	# convert dataframe into numpy array
	data = df.to_numpy()

	# separate features and target values
	x = data[:, :-1]
	y = data[:, -1]

	# normalise
	# x = (x - x.mean(axis = 0)) / x.std(axis = 0)

	return (x, y)

class LinearRegression():
	
	def __init__(self, x, y, t) :

		assert( type(x) == numpy.ndarray and x.ndim == 2 )
		assert( type(y) == numpy.ndarray and y.ndim == 1 )
		assert( x.shape[0] == y.shape[0] )
		assert( 0 <= t <= 1 )

		self.features = x
		self.target_values = y

		# test/train ratio 
		self.train_ratio = t
		self.test_ratio = 1 - t

		# test/train sets, to be calculated using self.split()
		self.train_set = None
		self.test_set = None

		# to store rmses for each epoch
		self.losses = None

		# model parameters after training
		self.fitted_parameters = None

		# to cehck if training is complete
		self.trained = False

		# to maintain testing error
		self.test_error = None

	def reset(self) :
		"""
		Reset all the progress made by model
		Input Parameters:
			None
		Return Values:
			None
		"""

		self.train_set = None
		self.test_set = None
		self.losses = None
		self.fitted_parameters = None
		self.trained = False
		self.test_error = None

	def get_splits(self) :
		"""
		Resturn the train and test datasets
		Input Parameters:
			None
		Return Values:
			(train_set, test_set) - a tuple containing training and testing datasets
		"""
		
		assert(self.train_set is not None)

		return ((self.train_set[0].copy(), self.train_set[1].copy()), (self.test_set[0].copy(), self.test_set[1].copy()))

	def split(self) :
		"""
		Splits the dataset into training and testing
		Input Parameters:
			None
		Return Values:
			None
		"""

		enteries_count = self.features.shape[0]
		train_size = int( enteries_count * self.train_ratio )

		indexes = numpy.arange(enteries_count)
		# numpy.random.seed(0)
		# numpy.random.shuffle(indexes)

		# split data into train/test sets
		self.train_set = (self.features[indexes[ : train_size]], self.target_values[indexes[ : train_size]])
		self.test_set = (self.features[indexes[train_size : ]], self.target_values[indexes[train_size : ]])

	def train(self, model_parameters = None, alpha = 0.1, epoch = 1000) :
		"""
		Train the model on the training set
		Input Parameters:
			model_parameters - initial value of model parameters
			alpha - learning rate
			epoch - number of iterations
		Return Values:
			None
		"""

		if model_parameters is None :
			model_parameters = numpy.zeros(self.features.shape[1] + 1)

		assert( type(model_parameters) == numpy.ndarray and model_parameters.ndim == 1 and model_parameters.shape[0] == self.features.shape[1] + 1 )
		assert( type(alpha) == int or type(alpha) == float )
		assert( type(epoch) == int )

		if self.train_set is None :
			self.split()

		self.losses = numpy.zeros(epoch)

		# extra column to take-care for intercept coefficient
		modified_train_set_features = numpy.hstack( (numpy.ones((self.train_set[0].shape[0], 1)), self.train_set[0]) )
		
		# Gradient Descent
		for e in range(epoch) :
			error, derivative = self.cost_and_derivative(modified_train_set_features, self.train_set[1], model_parameters)
			self.losses[e] = error

			model_parameters = model_parameters - alpha * derivative

		self.fitted_parameters = model_parameters
		self.trained = True

		return error

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
		dif = x.dot(m) - y

		error = ( ( 1 / n ) * numpy.sum( dif ** 2 ) ) ** 0.5
		derivative = ( 1 / n ) * ( x.T.dot(dif) ) / error

		return (error, derivative)

	def plot_loss(self) :
		"""
		Plot the Epoch vs RMSE graph for training
		Input Parameters:
			None
		Output:
			Plot of Epoch vs RMSE
		Return Values:
			None
		"""

		assert( self.trained == True )

		matplotlib.pyplot.plot(self.losses)
		matplotlib.pyplot.grid(True)
		matplotlib.pyplot.ylabel("Loss")
		matplotlib.pyplot.xlabel("Epoch")
		matplotlib.pyplot.rc('axes', labelsize = 20)
		matplotlib.pyplot.title("Epoch vs Loss", fontsize = 14)
		matplotlib.pyplot.savefig("Plots/Q1/1_1.jpg")
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
		if self.test_error is not None :
			return self.test_error 

		predicted_target_values = self.predict(self.test_set[0])

		test_rmse = ( ( 1 / self.test_set[1].shape[0] ) * numpy.sum( (self.test_set[1] - predicted_target_values) ** 2 )  ) ** 0.5

		return test_rmse

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

		y = x_modified.dot(self.fitted_parameters)

		return y

def CustomRegresion(train_set, test_set, alphas, model_type) :
	"""
	Calculate RMSE for given set using specified model
	Input Parameters:
		train_set - X, Y tuple for training data
		test_set - X, Y tuple for training data
		alphas - alpha on which to run the model
		model_type - model to use for running
	Return Values:
		rmses - rmse for all predictions
		coefficients - model coefficients for all alphas
	"""

	rmses = []
	coefficients = []
	for a in alphas :
		model = model_type(alpha = a)
		model.fit(train_set[0], train_set[1])

		y_actual = test_set[1]
		y_predicted = model.predict(test_set[0])
		n = y_actual.shape[0]

		rmse = ( ( 1 / n ) * numpy.sum( (y_actual - y_predicted ) ** 2 ) )  ** 0.5
		rmses.append(rmse)
		coefficients.append( numpy.hstack( (model.intercept_ , model.coef_)) )

	return (rmses, coefficients)

def plot_losses(rmse_ridge, rmse_lasso, alphas) :
	"""
	Plot RMSE for both Ridge and Lasso Model for given alphas
	Input Parameters:
		rmse_ridge - rmses values calculated using ridge model
		rmse_lasso - rmses values calculated using lasso model
		alphas - corresponding alphas for both rmse
	Output:
		Plot of Alphas vs RMSE
	Return Values:
		None
	"""

	matplotlib.pyplot.plot(alphas, rmse_ridge, marker = 'o', label = 'Ridge Regression')
	matplotlib.pyplot.plot(alphas, rmse_lasso, marker = 'o', label = 'Lasso Regression')
	matplotlib.pyplot.legend()
	matplotlib.pyplot.grid(True)
	matplotlib.pyplot.ylabel("RMSE")
	matplotlib.pyplot.xlabel("Alphas")
	matplotlib.pyplot.xscale('log')
	matplotlib.pyplot.rc('axes', labelsize = 20)
	matplotlib.pyplot.title("Alphas vs RMSE", fontsize = 14)
	matplotlib.pyplot.savefig("Plots/Q1/1_2a.jpg")
	matplotlib.pyplot.show()

def find_best_coefficients(rmses, coefficients, alphas) :
	"""
	Find minimum RMSE and the value of alpha
	Input Parameters:
		rmses - rmse values after running model
		coefficients - model coefficients for each alpha
		alphas - corresponding alphas for rmse
	Return Values:
		best_coefficients - (coefficient, alpha) tuple having minimum rmse among all and its corresponding alpha
	"""

	n = len(rmses)
	assert( n == len(alphas) )

	minIndex = 0
	for i in range(1, n) :
		if rmses[i] < rmses[minIndex] :
			minIndex = i

	best_coefficients = (coefficients[minIndex], alphas[minIndex])

	return best_coefficients

def find_best_alpha_and_coefficients(alphas, features, target, model_type) :
	"""
	Find the best alpha for given model and range
	Input Parameters:
		alphas - learning rates
		features - features of dataset
		dataset - target of dataset
		model_type - model for which to find best alpha
	Return Values:
		best_alpha - best value of alpha given model
		best_coefficients - best model coefficients of given model
	"""

	model = model_type()
	grid = sklearn.model_selection.GridSearchCV(estimator = model, param_grid = dict(alpha = alphas))
	grid.fit(features, target)

	best_alpha = grid.best_estimator_.alpha
	best_coefficients = numpy.hstack( (grid.best_estimator_.intercept_, grid.best_estimator_.coef_) )

	return (best_alpha, best_coefficients)

if __name__ == '__main__' :
	
	# 1.
	dataset_name_with_path = 'Datasets/Q1/abalone.data'
	
	x, y = preProccessData(dataset_name_with_path)
	
	model = LinearRegression(x.copy(), y.copy(), 0.8)
	model.split()
	train_rmse = model.train()
	model.plot_loss()
	test_rmse = model.test()

	print("Training Set RMSE:", train_rmse)
	print("Testing Set RMSE:", test_rmse)

	# 2. a
	train_set, test_set = model.get_splits()
	alphas = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 1e-1, 1, 2, 5, 10]

	rmses_ridge, coefficients_ridge = CustomRegresion( (train_set[0].copy(), train_set[1].copy()), (test_set[0].copy(), test_set[1].copy()), alphas.copy(), sklearn.linear_model.Ridge)
	rmses_lasso, coefficients_lasso = CustomRegresion( (train_set[0].copy(), train_set[1].copy()), (test_set[0].copy(), test_set[1].copy()), alphas.copy(), sklearn.linear_model.Lasso)

	plot_losses(rmses_ridge, rmses_lasso, alphas)
	best_coefficients_ridge = find_best_coefficients(rmses_ridge, coefficients_ridge, alphas.copy())
	best_coefficients_lasso = find_best_coefficients(rmses_lasso, coefficients_lasso, alphas.copy())

	print("Best model coefficients for Ridge", best_coefficients_ridge[0], "are obtained at", best_coefficients_ridge[1])
	print("Best model coefficients for Lasso", best_coefficients_lasso[0], "are obtained at", best_coefficients_lasso[1])

	# 2. b
	best_alpha_ridge, best_coefficients_ridge = find_best_alpha_and_coefficients(alphas.copy(), x.copy(), y.copy(), sklearn.linear_model.Ridge)
	best_alpha_lasso, best_coefficients_lasso = find_best_alpha_and_coefficients(alphas.copy(), x.copy(), y.copy(), sklearn.linear_model.Lasso)

	print("Best Alpha for Ridge:", best_alpha_ridge)
	print("Best Alpha for Lasso:", best_alpha_lasso)
	print("Best Coefficients for Ridge:", best_coefficients_ridge)
	print("Best Coefficients for Lasso:", best_coefficients_lasso)
