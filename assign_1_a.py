import numpy as np
import pandas as pd
import os
import sys

#location - C:\PG\Semester 5\COL341 - Machine Learning\Assignment 1\col341_a1\2016MT10628

x_train_path = sys.argv[1]
x_path = os.path.abspath(x_train_path)
path_train = os.path.dirname(x_path)
os.chdir(path_train)

x_train = pd.read_csv(x_train_path, header = None, na_filter = False, dtype = np.float64, low_memory = False)

x_test_path = sys.argv[2]
actual_path_x_test = os.path.abspath(x_test_path)
path_test = os.path.dirname(actual_path_x_test)
os.chdir(path_test)

x_test = pd.read_csv(x_test_path, header = None, na_filter = False, dtype = np.float64, low_memory = False)

x_train = x_train.values
x_test = x_test.values
y_test = list(x_test[:,0])
y_train = list(x_train[:,0])
x_test[:,0] = 1
x_train[:,0] = 1

def predict(x_train, y_train, x_test, y_test):
	inv1 = np.matmul(np.transpose(x_train),x_train)
	inv2 = np.linalg.inv(inv1)
	inv3 = np.matmul(np.transpose(x_train),y_train)
	coeff = np.matmul(inv2,inv3)
	y_out = np.matmul(x_test,coeff)
	return y_out

x = predict(x_train, y_train, x_test, y_test)

x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)
np.savetxt(x_output, x)

#val = min(y_test)
#print('Linear Regression')
#print(np.linalg.norm(y_test-x)**2/np.linalg.norm(y_test-val)**2)