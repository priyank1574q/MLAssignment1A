# COL341: Assignment 1(A)
## Linear Regression

**Name**

linreg - Run the executable program for linear regression

**Synopsis**

`./linreg <part> <tr> <ts> <out>`

**Description**

This program will train linear regression model using given code on train data, make predictions on test data and write final predictions
in given output file.

**Options**

-part  
    Part as per question i.e. a/b/c.  
-tr  
    File containing training data in csv format where 1st entry is the target  
-ts  
    File containing test data in csv format where 1st entry is the target  
-out  
    Output file for predictions. One value in each line.

**Example**
    
`./linreg a train.csv test.csv output`
    
**Data**

- msd_train.csv: Train data  
- msd_test.csv: Test data

**Checking Program**

Normalized mean squared error will be used as an evaluation criterion:
    error = \frac{\sum_{i=1}^{m} (y_i - \hat{y_i})^2}{\sum_{i=1}^{m} (y_i - min\_val)^2}
