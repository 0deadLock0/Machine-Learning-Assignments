
import pandas
import numpy
import sklearn.tree
import matplotlib.pyplot
import sklearn.ensemble


def preproccess_dataset(file_name_with_relative_path):
	"""
	Preproccess the raw data for easy/better use
	Input Paramteres :
		file_name_with_relative_path - name of file from which contains data
	Return Values :
		x - features
		y - target value
	"""

	df = pandas.read_csv(file_name_with_relative_path)

	# remove useless columns
	df.drop('No', axis = 1, inplace = True)

	# for better understanding of dataset
	# print(df.head())
	# print(df.info())
	# with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
	# 	print(df.describe())
	# print(df['cbwd'].describe())

	# fixing null values
	null_headers = ["pm2.5"]
	for h in null_headers :
		df[h].fillna(df[h].median(), inplace = True)
	# print(df.info())

	# comverting string values into numerical values
	uniqueValues = df['cbwd'].unique()
	# print(uniqueValues)
	replacement_mapping_dict = {"NW" : 1, "NE" : 2, "SE" : 3, "cv" : 4}
	df['cbwd'].replace(replacement_mapping_dict, inplace = True)
	# print(df['cbwd'].describe())

	# shufle rows
	df = df.sample(frac = 1).reset_index(drop = True)

	# convert dataframe into numpy array
	y = df['month'].to_numpy()
	x = df.drop('month', axis = 1).to_numpy()
	# print(x, y)

	return (x, y)

def split_dataset(features, target, train_ratio, val_ratio, test_ratio) :
	"""
	Splits the dataset into training, validation and testing
	Input Parameters:
		features - features columns of dataset
		target - target columns of dataset
		train_ratio - the ratio of split for training set
		val_ratio - the ratio of split for validation set
		test_ratio - the ratio of split for testing set
	Return Values:
		train_set - a two tuple containing features and target of training set
		val_set - a two tuple containing features and target of validation set
		test_set - a two tuple containing features and target of testing set
	"""

	assert(type(features) == numpy.ndarray and features.ndim >= 1 and features.ndim <= 2 and type(target) == numpy.ndarray and target.ndim == 1 and features.shape[0] == target.shape[0])
	assert(train_ratio + val_ratio + test_ratio == 1)

	enteries_count = features.shape[0]
	train_size = int( enteries_count * train_ratio )
	val_size = int( enteries_count * val_ratio )

	val_index_l = train_size
	val_index_r = train_size + val_size

	indexes = numpy.arange(enteries_count)
	numpy.random.shuffle(indexes)

	# split data into train/test sets
	train_set = (features[indexes[ : train_size]], target[indexes[ : train_size]])
	val_set = (features[indexes[ val_index_l : val_index_r ]], target[indexes[ val_index_l : val_index_r ]])
	test_set = (features[indexes[ val_index_r : ]], target[indexes[ val_index_r : ]])

	# print(features.shape[0], train_set[0].shape[0], val_set[0].shape[0], test_set[0].shape[0])

	return (train_set, val_set, test_set)

def get_accuracy(y_true, y_predicted) :
	"""
	Calculates the accuracy from given true and predicted values
	Input Parameters:
		y_true - true target values
		y_predicted - predicted target values
	Output Parameters:
		Accuracy - computed accuracy
	"""

	assert( type(y_true) == numpy.ndarray and y_true.ndim == 1 )
	assert( type(y_predicted) == numpy.ndarray and y_predicted.ndim == 1 )
	assert( y_true.shape[0] == y_predicted.shape[0] )

	c = max(numpy.unique(y_true)) + 1
	confusion_matrix = numpy.zeros((c, c), dtype = int)

	for i in range(y_true.shape[0]):
		confusion_matrix[y_true[i]][y_predicted[i]] += 1

	accuracy = confusion_matrix.trace() / y_true.shape[0]

	return accuracy

def fit_model_and_get_accuracy(train_set, test_set, criterion, max_depth = None, val_set = None) :
	"""
	Fit the DecisionTree on given parameters and return the accuracy on train, val and test set
	Input Parameters:
		train_set - a two tuple containing features and target of training set
		val_set - a two tuple containing features and target of validation set
		test_set - a two tuple containing features and target of testing set
		criterion - criterion to use in model ('gini'/'entropy')
		max_depth - maximum allowable depth of tree
	Output Parameters:
		train_accuracy - accuracy of model on training set
		test_accuracy - accuracy of model on testing set
		val_accuracy[optional] - accuracy of model on validation set
	"""

	assert( criterion == 'gini' or criterion == 'entropy' )
	assert( max_depth is None or ( type(max_depth) == int and max_depth > 0 ) )

	assert( (type(train_set) == tuple or type(train_set) == list) and len(train_set) == 2 )
	assert( type(train_set[0]) == numpy.ndarray and train_set[0].ndim >= 1 and train_set[0].ndim <= 2 and type(train_set[1]) == numpy.ndarray and train_set[1].ndim == 1 )
	assert( train_set[0].shape[0] == train_set[1].shape[0] )

	assert( (type(test_set) == tuple or type(test_set) == list) and len(test_set) == 2 )
	assert( type(test_set[0]) == numpy.ndarray and test_set[0].ndim >= 1 and test_set[0].ndim <= 2 and type(test_set[1]) == numpy.ndarray and test_set[1].ndim == 1 )
	assert( test_set[0].shape[0] == test_set[1].shape[0] )

	assert( val_set is None or ( (type(val_set) == tuple or type(val_set) == list) and len(val_set) == 2 and type(val_set[0]) == numpy.ndarray and val_set[0].ndim >= 1 and val_set[0].ndim <= 2 and type(val_set[1]) == numpy.ndarray and val_set[1].ndim == 1 and val_set[0].shape[0] == val_set[1].shape[0] ) )

	decision_tree = sklearn.tree.DecisionTreeClassifier(criterion = criterion, max_depth = max_depth)
	decision_tree = decision_tree.fit(train_set[0], train_set[1])

	train_accuracy = get_accuracy(train_set[1], decision_tree.predict(train_set[0]))
	test_accuracy = get_accuracy(test_set[1], decision_tree.predict(test_set[0]))
	val_accuracy = None if val_set is None else get_accuracy(val_set[1], decision_tree.predict(val_set[0]))

	return (train_accuracy, test_accuracy) if val_accuracy is None else (train_accuracy, test_accuracy, val_accuracy)

def ensemble_and_compute_accuracy(number_of_stumps, criterion, max_depth, ratio_to_train, classes, train_set, test_set, val_set = None) :
	"""
	Peform Ensembling by creating multiple DecisionTrees on given parameters and prints the accuracy on train, val and test set
	Input Parameters:
		train_set - a two tuple containing features and target of training set
		val_set - a two tuple containing features and target of validation set
		test_set - a two tuple containing features and target of testing set
		number_of_stumps - number of stumps in decision tree
		criterion - criterion to use in model ('gini'/'entropy')
		max_depth - maximum allowable depth of tree
		ratio_to_train - ratio of training data to be used for model fiting
		classes - number of classes in the target column
	Output Parameters:
		None
	"""

	assert( type(number_of_stumps) == int and number_of_stumps >= 1)
	assert( criterion == 'gini' or criterion == 'entropy' )
	assert( max_depth is None or ( type(max_depth) == int and max_depth > 0 ) )
	assert( ratio_to_train > 0 and ratio_to_train <= 1 )

	assert( (type(train_set) == tuple or type(train_set) == list) and len(train_set) == 2 )
	assert( type(train_set[0]) == numpy.ndarray and train_set[0].ndim >= 1 and train_set[0].ndim <= 2 and type(train_set[1]) == numpy.ndarray and train_set[1].ndim == 1 )
	assert( train_set[0].shape[0] == train_set[1].shape[0] )

	assert( (type(test_set) == tuple or type(test_set) == list) and len(test_set) == 2 )
	assert( type(test_set[0]) == numpy.ndarray and test_set[0].ndim >= 1 and test_set[0].ndim <= 2 and type(test_set[1]) == numpy.ndarray and test_set[1].ndim == 1 )
	assert( test_set[0].shape[0] == test_set[1].shape[0] )

	assert( val_set is None or ( (type(val_set) == tuple or type(val_set) == list) and len(val_set) == 2 and type(val_set[0]) == numpy.ndarray and val_set[0].ndim >= 1 and val_set[0].ndim <= 2 and type(val_set[1]) == numpy.ndarray and val_set[1].ndim == 1 and val_set[0].shape[0] == val_set[1].shape[0] ) )

	def update_counts(overall, current):
		"""
		Updates the occurences in the overall according to values in current
		Input Parameters:
			overall - in which updates are to be made 
			current - according to which updates are to be made
		Output Parameters:
			None
		"""

		assert( type(overall) == numpy.ndarray and overall.ndim == 2 )
		assert( type(current) == numpy.ndarray and current.ndim == 1 )
		assert( overall.shape[0] == current.shape[0] )

		n = overall.shape[0]
		for i in range(n) :
			overall[i][current[i]] += 1

	train_predictions_count = numpy.zeros((train_set[0].shape[0], classes + 1), dtype = int)
	test_predictions_count = numpy.zeros((test_set[0].shape[0], classes + 1), dtype = int)
	val_predictions_count = None if val_set is None else numpy.zeros((val_set[0].shape[0], classes + 1), dtype = int)

	to_train_size = int(train_set[0].shape[0] * ratio_to_train)

	for i in range(number_of_stumps) :
		decision_tree = sklearn.tree.DecisionTreeClassifier(criterion = criterion, max_depth = max_depth)
		
		selected_indexes = numpy.random.choice(train_set[0].shape[0], to_train_size, replace=False)
		filtered_train_set_feature = train_set[0][selected_indexes]
		filtered_train_set_target = train_set[1][selected_indexes]

		decision_tree = decision_tree.fit(filtered_train_set_feature, filtered_train_set_target)

		train_predictions = decision_tree.predict(train_set[0])
		test_predictions = decision_tree.predict(test_set[0])
		val_predictions = None if val_set is None else decision_tree.predict(val_set[0])

		update_counts(train_predictions_count, train_predictions)
		update_counts(test_predictions_count, test_predictions)
		if val_predictions is not None:
			update_counts(val_predictions_count, val_predictions)
	
	overall_train_predictions = numpy.argmax(train_predictions_count, axis = 1)
	overall_test_predictions = numpy.argmax(test_predictions_count, axis = 1)
	overall_val_predictions = None if val_set is None else numpy.argmax(val_predictions_count, axis = 1)

	train_accuracy = get_accuracy(train_set[1], overall_train_predictions)
	test_accuracy = get_accuracy(test_set[1], overall_test_predictions)
	val_accuracy = None if val_set is None else get_accuracy(val_set[1], overall_val_predictions)

	print("For Decision Stumps =", number_of_stumps, ", Maximum Depth =", max_depth, ", Used training set ratio =", ratio_to_train, ":-")
	print("Training Accuracy =", train_accuracy, end = " ")
	if val_set is not None :
		print("Validation Accuracy =", val_accuracy, end = " ")
	print("Testing Accuracy =", test_accuracy, end = " ")
	print()

def part_a(train_set, val_set, test_set) :
	"""
	Performs the part a, of the assignment
	Input Parameters:
		train_set - a two tuple containing features and target of training set
		val_set - a two tuple containing features and target of validation set
		test_set - a two tuple containing features and target of testing set
	Output Parameters:
		criterion - the criterion ('gini' and 'entropy') on which model performs the best
	"""

	assert( (type(train_set) == tuple or type(train_set) == list) and len(train_set) == 2 )
	assert( type(train_set[0]) == numpy.ndarray and train_set[0].ndim >= 1 and train_set[0].ndim <= 2 and type(train_set[1]) == numpy.ndarray and train_set[1].ndim == 1 )
	assert( train_set[0].shape[0] == train_set[1].shape[0] )

	assert( (type(test_set) == tuple or type(test_set) == list) and len(test_set) == 2 )
	assert( type(test_set[0]) == numpy.ndarray and test_set[0].ndim >= 1 and test_set[0].ndim <= 2 and type(test_set[1]) == numpy.ndarray and test_set[1].ndim == 1 )
	assert( test_set[0].shape[0] == test_set[1].shape[0] )

	assert( (type(val_set) == tuple or type(val_set) == list) and len(val_set) == 2 )
	assert( type(val_set[0]) == numpy.ndarray and val_set[0].ndim >= 1 and val_set[0].ndim <= 2 and type(val_set[1]) == numpy.ndarray and val_set[1].ndim == 1 )
	assert( val_set[0].shape[0] == val_set[1].shape[0] )

	accuracies_gini = fit_model_and_get_accuracy(train_set, test_set, 'gini')
	accuracies_entropy = fit_model_and_get_accuracy(train_set, test_set, 'entropy')

	# print('gini:', accuracies_gini)
	# print('entropy', accuracies_entropy)

	criterion, best_accuracy = ('gini', accuracies_gini[1]) if accuracies_gini[1] > accuracies_entropy[1] else ('entropy', accuracies_entropy[1])
	print(criterion, "gives better accuray on the testing set, with a performance of", best_accuracy)

	return criterion

def part_b(criterion, train_set, val_set, test_set) :
	"""
	Performs the part b, of the assignment
	Input Parameters:
		criterion - the criterion used to evaluate splits of tree
		train_set - a two tuple containing features and target of training set
		val_set - a two tuple containing features and target of validation set
		test_set - a two tuple containing features and target of testing set
	Output Parameters:
		best_depth - the depth on which model performs the best
	"""

	assert( criterion == 'gini' or criterion == 'entropy' )

	assert( (type(train_set) == tuple or type(train_set) == list) and len(train_set) == 2 )
	assert( type(train_set[0]) == numpy.ndarray and train_set[0].ndim >= 1 and train_set[0].ndim <= 2 and type(train_set[1]) == numpy.ndarray and train_set[1].ndim == 1 )
	assert( train_set[0].shape[0] == train_set[1].shape[0] )

	assert( (type(test_set) == tuple or type(test_set) == list) and len(test_set) == 2 )
	assert( type(test_set[0]) == numpy.ndarray and test_set[0].ndim >= 1 and test_set[0].ndim <= 2 and type(test_set[1]) == numpy.ndarray and test_set[1].ndim == 1 )
	assert( test_set[0].shape[0] == test_set[1].shape[0] )

	assert( (type(val_set) == tuple or type(val_set) == list) and len(val_set) == 2 )
	assert( type(val_set[0]) == numpy.ndarray and val_set[0].ndim >= 1 and val_set[0].ndim <= 2 and type(val_set[1]) == numpy.ndarray and val_set[1].ndim == 1 )
	assert( val_set[0].shape[0] == val_set[1].shape[0] )

	depths = [2, 4, 8, 10, 15, 30]
	training_accuracies = []
	testing_accuracies = []

	for d in depths :
		model_accuracies = fit_model_and_get_accuracy(train_set, test_set, criterion, d)
		training_accuracies.append(model_accuracies[0])
		testing_accuracies.append(model_accuracies[1])

	matplotlib.pyplot.plot(depths, training_accuracies, marker = 'o', label = 'Training')
	matplotlib.pyplot.plot(depths, testing_accuracies, marker = 'x', label = 'Testing')
	matplotlib.pyplot.legend()
	matplotlib.pyplot.grid(True)
	matplotlib.pyplot.ylabel("Accuracy")
	matplotlib.pyplot.xlabel("Depths")
	matplotlib.pyplot.rc('axes', labelsize = 20)
	matplotlib.pyplot.title("Training and Testing Accuracy for different Depths", fontsize = 14)
	matplotlib.pyplot.savefig("Plots/Q1/b.jpg")
	matplotlib.pyplot.show()

	best_depth_index = 0
	n = len(depths)
	for i in range(1, n) : 
		if testing_accuracies[i] > testing_accuracies[best_depth_index] : 
			best_depth_index = i;
		elif testing_accuracies[i] == testing_accuracies[best_depth_index] :
			if training_accuracies[i] > training_accuracies[best_depth_index] :
				best_depth_index = i
	best_depth = depths[best_depth_index]

	print("At depth =", best_depth, "maximum performance is observed.", "Training Score:", training_accuracies[best_depth_index], "Testing Score:", testing_accuracies[best_depth_index])

	return best_depth

def part_c(criterion, classes, train_set, val_set, test_set) :
	"""
	Performs the part b, of the assignment
	Input Parameters:
		criterion - the criterion used to evaluate splits of tree
		classes - number of classes in target
		train_set - a two tuple containing features and target of training set
		val_set - a two tuple containing features and target of validation set
		test_set - a two tuple containing features and target of testing set
	Output Parameters:
		None
	"""
	
	assert( criterion == 'gini' or criterion == 'entropy' )
	assert( type(classes) == int and classes > 0 )
	
	assert( (type(train_set) == tuple or type(train_set) == list) and len(train_set) == 2 )
	assert( type(train_set[0]) == numpy.ndarray and train_set[0].ndim >= 1 and train_set[0].ndim <= 2 and type(train_set[1]) == numpy.ndarray and train_set[1].ndim == 1 )
	assert( train_set[0].shape[0] == train_set[1].shape[0] )

	assert( (type(test_set) == tuple or type(test_set) == list) and len(test_set) == 2 )
	assert( type(test_set[0]) == numpy.ndarray and test_set[0].ndim >= 1 and test_set[0].ndim <= 2 and type(test_set[1]) == numpy.ndarray and test_set[1].ndim == 1 )
	assert( test_set[0].shape[0] == test_set[1].shape[0] )

	assert( (type(val_set) == tuple or type(val_set) == list) and len(val_set) == 2 )
	assert( type(val_set[0]) == numpy.ndarray and val_set[0].ndim >= 1 and val_set[0].ndim <= 2 and type(val_set[1]) == numpy.ndarray and val_set[1].ndim == 1 )
	assert( val_set[0].shape[0] == val_set[1].shape[0] )

	number_of_stumps = 100
	max_depth = 3
	ratio_to_train = 0.5
	ensemble_and_compute_accuracy(number_of_stumps, criterion, max_depth, ratio_to_train, classes, train_set, test_set)

def part_d(criterion, classes, best_depth, train_set, val_set, test_set) :
	"""
	Performs the part b, of the assignment
	Input Parameters:
		criterion - the criterion used to evaluate splits of tree
		classes - number of classes in target
		best_depth - the maximum allowable depth of tree
		train_set - a two tuple containing features and target of training set
		val_set - a two tuple containing features and target of validation set
		test_set - a two tuple containing features and target of testing set
	Output Parameters:
		None
	"""
	
	assert( criterion == 'gini' or criterion == 'entropy' )
	assert( type(classes) == int and classes > 0 )
	assert( type(best_depth) == int and best_depth > 0 )

	assert( (type(train_set) == tuple or type(train_set) == list) and len(train_set) == 2 )
	assert( type(train_set[0]) == numpy.ndarray and train_set[0].ndim >= 1 and train_set[0].ndim <= 2 and type(train_set[1]) == numpy.ndarray and train_set[1].ndim == 1 )
	assert( train_set[0].shape[0] == train_set[1].shape[0] )

	assert( (type(test_set) == tuple or type(test_set) == list) and len(test_set) == 2 )
	assert( type(test_set[0]) == numpy.ndarray and test_set[0].ndim >= 1 and test_set[0].ndim <= 2 and type(test_set[1]) == numpy.ndarray and test_set[1].ndim == 1 )
	assert( test_set[0].shape[0] == test_set[1].shape[0] )

	assert( (type(val_set) == tuple or type(val_set) == list) and len(val_set) == 2 )
	assert( type(val_set[0]) == numpy.ndarray and val_set[0].ndim >= 1 and val_set[0].ndim <= 2 and type(val_set[1]) == numpy.ndarray and val_set[1].ndim == 1 )
	assert( val_set[0].shape[0] == val_set[1].shape[0] )

	number_of_stumps = [10, 25, 50, 100]
	max_depths = [4, 8, 10, 15, 20] + [best_depth]
	ratio_to_train = 0.5

	for s in number_of_stumps :
		for d in max_depths :
			ensemble_and_compute_accuracy(s, criterion, d, ratio_to_train, classes, train_set, test_set, val_set)

def part_e(criterion, depth, train_set, val_set, test_set) :
	"""
	Performs the part a, of the assignment
	Input Parameters:
		criterion - the criterion used to evaluate splits of tree
		depth - the maximum allowable depth of tree
		train_set - a two tuple containing features and target of training set
		val_set - a two tuple containing features and target of validation set
		test_set - a two tuple containing features and target of testing set
	Output Parameters:
		None
	"""

	assert( criterion == 'gini' or criterion == 'entropy' )
	assert( type(depth) == int and depth > 0 )

	assert( (type(train_set) == tuple or type(train_set) == list) and len(train_set) == 2 )
	assert( type(train_set[0]) == numpy.ndarray and train_set[0].ndim >= 1 and train_set[0].ndim <= 2 and type(train_set[1]) == numpy.ndarray and train_set[1].ndim == 1 )
	assert( train_set[0].shape[0] == train_set[1].shape[0] )

	assert( (type(test_set) == tuple or type(test_set) == list) and len(test_set) == 2 )
	assert( type(test_set[0]) == numpy.ndarray and test_set[0].ndim >= 1 and test_set[0].ndim <= 2 and type(test_set[1]) == numpy.ndarray and test_set[1].ndim == 1 )
	assert( test_set[0].shape[0] == test_set[1].shape[0] )

	assert( (type(val_set) == tuple or type(val_set) == list) and len(val_set) == 2 )
	assert( type(val_set[0]) == numpy.ndarray and val_set[0].ndim >= 1 and val_set[0].ndim <= 2 and type(val_set[1]) == numpy.ndarray and val_set[1].ndim == 1 )
	assert( val_set[0].shape[0] == val_set[1].shape[0] )

	number_estimators = [4, 8, 10, 15, 20]
	for e in number_estimators :
		adaboost_model = sklearn.ensemble.AdaBoostClassifier(base_estimator = sklearn.tree.DecisionTreeClassifier(criterion = criterion, max_depth = depth), n_estimators = e)
		adaboost_model = adaboost_model.fit(train_set[0], train_set[1])

		# train_accuracy = get_accuracy(train_set[1], adaboost_model.predict(train_set[0]))
		# val_accuracy = get_accuracy(val_set[1], adaboost_model.predict(val_set[0]))
		test_accuracy = get_accuracy(test_set[1], adaboost_model.predict(test_set[0]))
		
		print("For Number of Estimator =", e, ", Criterion =", criterion, ", Maximum Depth =", depth, ":-")
		# print("Training Accuracy =", train_accuracy, end = " ")
		# print("Validation Accuracy =", val_accuracy, end = " ")
		print("Testing Accuracy =", test_accuracy, end = " ")
		print()

if __name__ == '__main__' :
	
	numpy.random.seed(0)

	# Dataset loading, preproccessing, spliting 
	dataset_name_with_path = 'Datasets/Q1/PRSA_data_2010.1.1-2014.12.31.csv'
	x, y = preproccess_dataset(dataset_name_with_path)
	train_set, val_set, test_set = split_dataset(x, y, 0.7, 0.15, 0.15)

	# a.
	print("\nPart A :")
	best_criterion = part_a(train_set, val_set, test_set)

	# b.
	print("\nPart B :")
	best_depth = part_b(best_criterion, train_set, val_set, test_set)

	classes = 12 # number of months

	# c.
	print("\nPart C :")
	part_c(best_criterion, classes, train_set, val_set, test_set)

	# d.
	print("\nPart D :")
	part_d(best_criterion, classes, best_depth, train_set, val_set, test_set)

	# e.
	print("\nPart E :")
	part_e(best_criterion, best_depth, train_set, val_set, test_set)
