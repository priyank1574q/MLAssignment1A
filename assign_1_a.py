# Importing required libraries
import numpy as np
import pandas as pd
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
x_test = pd.read_csv(x_test_path, header = None, na_filter = False, dtype = np.float64, low_memory = False)

# Converting dataframe to numpy array
x_train = x_train.values
x_test = x_test.values

# First column contains the output values
y_test = list(x_test[:,0])
y_train = list(x_train[:,0])

# Setting first column as bias term
x_test[:,0] = 1
x_train[:,0] = 1

# Using normal equations for predicting on test data
def predict(x_train, y_train, x_test, y_test):
	inv1 = np.matmul(np.transpose(x_train),x_train)
	inv2 = np.linalg.inv(inv1)
	inv3 = np.matmul(np.transpose(x_train),y_train)
	coeff = np.matmul(inv2,inv3)
	y_out = np.matmul(x_test,coeff)
	return y_out

# Predicting on test data
x = predict(x_train, y_train, x_test, y_test)

# Saving predictions on an output file
x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)
np.savetxt(x_output, x)
