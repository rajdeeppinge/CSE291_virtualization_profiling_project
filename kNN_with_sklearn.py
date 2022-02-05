import csv
import numpy as np
from memory_profiler import profile


train_data = None
test_data = None
train_X = None
train_y = None
test_X = None
test_y = None

model = None

@profile
def read_csv_file(filename):
    with open(filename, 'r') as csvfile:
        csv_data = csv.reader(csvfile, delimiter=',')
        
        list_data = []
        
        for row in csv_data:
            list_data.append(row)
            
        return list_data


@profile
def load_data():
	global train_data
	global test_data

	data_dir = "MNIST_dataset_csv"
	
	train_data_filename = data_dir + '/mnist_train.csv'
	train_data = read_csv_file(train_data_filename)

	test_data_filename = data_dir + '/mnist_test.csv'
	test_data = read_csv_file(test_data_filename)



@profile
def prepare_data():
	global train_data
	global test_data

	global train_X
	global train_y
	global test_X
	global test_y


	train_data = train_data[1:] # remove titles from data
	test_data = test_data[1:] # remove titles from data

	# convert to 2d array with integer data type to convert string entries from csv to int
	train_data_2d_array = np.array(train_data, dtype='i')
	test_data_2d_array = np.array(test_data, dtype='i')

	# print shapes of 2d array for verification
	print(train_data_2d_array.shape)
	print(test_data_2d_array.shape)



	# Split label and pixel values for training data

	train_y = train_data_2d_array[:, 0]
	#train_y = np.reshape(train_y, (1, train_y.shape[0])) # converting to row vector as sklearn expects a row vector
	print(train_y.shape)

	train_X = train_data_2d_array[:, 1:]
	print(train_X.shape)



	# Split label and pixel values for test data

	test_y = test_data_2d_array[:, 0]
	#test_y = np.reshape(test_y, (1, test_y.shape[0]))
	print(test_y.shape)

	test_X = test_data_2d_array[:, 1:]
	print(test_X.shape)


@profile
def train_model():
	global train_X
	global train_y
	global model

	# create model and fit with training data
	from sklearn.neighbors import KNeighborsClassifier

	model = KNeighborsClassifier(n_neighbors=1)

	model.fit(train_X, train_y)


@profile
def prediction():
	global test_X
	global test_y
	global model

	# make prediction on test data
	predicted_test_y = model.predict(test_X)


	# display accuracy metrics
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import classification_report
	from sklearn.metrics import confusion_matrix

	accuracy = accuracy_score(test_y, predicted_test_y)
	error_rate = 1 - accuracy
	print("Accuracy = {0}, error_rate = {1}".format(accuracy, round(error_rate, 4)))
	print()
	print("Classification report")
	print(classification_report(test_y, predicted_test_y))
	print()
	print("Confusion matrix")
	print(confusion_matrix(test_y, predicted_test_y))



if __name__=="__main__":
	load_data()
	prepare_data()
	train_model()
	prediction()
