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

x_test = pd.read_csv('msd_test.csv', header = None, na_filter = False, dtype = np.float64, low_memory = False)

x_train = x_train.values
x_test = x_test.values
y_test = list(x_test[:,0])
y_train = list(x_train[:,0])
x_test[:,0] = 1
x_train[:,0] = 1

def predict(x_train, y_train, x_test, y_test, l):
	iden = np.identity(91)
	iden[0,0] = 0
	inv1 = np.matmul(np.transpose(x_train),x_train) + l*iden
	inv2 = np.linalg.inv(inv1)
	inv3 = np.matmul(np.transpose(x_train),y_train)
	coeff = np.matmul(inv2,inv3)
	y_out = np.matmul(x_test,coeff)
	return y_out

def linear_err(x_train, y_train, x_test, y_test, l):
	iden = np.identity(91)
	iden[0,0] = 0
	inv1 = np.matmul(np.transpose(x_train),x_train) + l*iden
	inv2 = np.linalg.inv(inv1)
	inv3 = np.matmul(np.transpose(x_train),y_train)
	coeff = np.matmul(inv2,inv3)
	y_out = np.matmul(x_test,coeff)
	for i in range(len(y_out)):
		y_out[i] = round(y_out[i])
	val = min(y_test)
	return (np.linalg.norm(y_test-y_out)**2/np.linalg.norm(y_test-val)**2)

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
		accuracy.append(linear_err(train_x, test_y, x[i], y[i], l))
	acc = 0
	for i in range(len(accuracy)):
		acc += accuracy[i]
	return (acc/len(accuracy))

#l = []
#for i in range(58220,58235,1):
#	l.append(i)

#for i in range(len(l)):
#	print('Lamda is - ' + str(l[i]))
#	print('Error is - ' + str(cross_acc(x_train, y_train, l[i])))
#	print('-----------------------------')

x = predict(x_train, y_train, x_test, y_test, 58226.5)

x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)
np.savetxt(x_output, x)

#min_val = min(y_test)
#print(np.linalg.norm(y_test-x)**2/np.linalg.norm(y_test-min_val)**2)