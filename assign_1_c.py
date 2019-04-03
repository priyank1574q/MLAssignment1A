# Importing required libraries
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import os
import sys

# Train data filepath
x_train_path = sys.argv[1]
x_path = os.path.abspath(x_train_path)
path_train = os.path.dirname(x_path)
os.chdir(path_train)

# Importing training data
x_train = pd.read_csv(x_train_path, header = None, na_filter = False, dtype = np.float64, low_memory = False)

# Test data filepath
x_test_path = sys.argv[2]
actual_path_x_test = os.path.abspath(x_test_path)
path_test = os.path.dirname(actual_path_x_test)
os.chdir(path_test)

# Importing testing data
x_test = pd.read_csv('msd_test.csv', header = None, na_filter = False, dtype = np.float64, low_memory = False)

# Converting dataframe to numpy array
x_train = x_train.values
x_test = x_test.values

# First column contains the output values
y_test = list(x_test[:,0])
y_train = list(x_train[:,0])

# Setting first column as bias term
x_test[:,0] = 1
x_train[:,0] = 1

# Non-correlated columns in dataset, using them for feature creation
imp = [1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 76, 85, 88]

# Adding sequential product of each of those columns in dataset
# Line (1,2), (2,3), (3,4)... and not (1,3), (1,4)...
for i in range(len(imp)):
	feature_train = x_train[:,imp[i]]*x_train[:,imp[(i+1)%len(imp)]]
	feature_test = x_test[:,imp[i]]*x_test[:,imp[(i+1)%len(imp)]]	
	x_train = np.column_stack((x_train, feature_train))
	x_test = np.column_stack((x_test, feature_test))

# New un-correlated column, adding it to dataset
imp.append(104)

# Adding square of each column
for i in range(len(imp)):
	feature_train = x_train[:,imp[i]]**2
	feature_test = x_test[:,imp[i]]**2
	x_train = np.column_stack((x_train, feature_train))
	x_test = np.column_stack((x_test, feature_test))

def divider(x_train, y_train):
	x_cross = []
	y_cross = []
	step = round((x_train.shape[0])/10)
	for i in range(0,x_train.shape[0],step):
		x_cross.append(x_train[i:i+step])
	for i in range(0,x_train.shape[0],step):
		y_cross.append(y_train[i:i+step])
	return x_cross, y_cross

def cross_acc(x_train, y_train, l):
	accuracy = []
	x, y = divider(x_train, y_train)
	for i in range(10):
		train = list(x)
		test = list(y)
		del train[i]
		del test[i]
		for j in range(8):
			train[j+1] = np.concatenate((train[j], train[j+1]), axis = 0)
			test[j+1] = np.concatenate((test[j], test[j+1]), axis = 0)
		train_x = train[8]
		test_y = test[8]
		reg = lm.LassoLars(alpha = l, normalize = True, fit_intercept = True, precompute = 'auto', max_iter = 1000, eps = 0.1, copy_X = True, fit_path = True, positive=False)
		reg.fit(train_x,test_y)
		val = min(y[i])
		accuracy.append(np.linalg.norm(reg.predict(x[i])-y[i])**2/np.linalg.norm(y[i]-val)**2)
	acc = 0
	for i in range(len(accuracy)):
		acc += accuracy[i]
	return (acc/len(accuracy))

#l = [3*10**(-7), 4*10**(-7), 5*10**(-7), 6*10**(-7), 7*10**(-7)]
l = 5*10**(-7)

#for i in range(len(l)):
#	print('Lamda is - ' + str(l[i]))
#	print('Error is - ' + str(cross_acc(x_train, y_train, l[i])))
#	print('-----------------------------')

reg = lm.LassoLars(alpha = l, normalize = True, fit_intercept = True, precompute = 'auto', max_iter = 1000, eps = 0.1, copy_X = True, fit_path = True, positive=False)
reg.fit(x_train,y_train)
x = reg.predict(x_test)

x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)
np.savetxt(x_output, x)

#val = min(y_test)
#print(np.linalg.norm(y_test-x)**2/np.linalg.norm(y_test-val)**2)
