import numpy as np
import pandas as pd
import math
from scipy.stats import norm

import tensorflow.compat.v1 as tf
from gain import gain

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from statsmodels.stats.contingency_tables import mcnemar

# Create a function for data generation
def datagen(n, cen):
    # Create random data according to the distributions
    x1 = np.random.normal(size = n, loc = 0, scale = 1)
    x2 = np.random.uniform(size = n, low = 0, high = 1)   
    x3 = np.random.normal(size = n, loc = 0, scale = 1)
    x4 = np.random.binomial(size = n, n = 1, p = 0.5)    # Bernoulli(0.5) = Binomial(1, 0.5)
    
    # We set the true coefficients to be b0 = b1 = b2 = b3 = b4 = 1
    # We generate the response variable y by the following
    logit = x1 + x2 + x3 + x4 + 1
    prob = 1 / (1 + math.e**(-logit))    
    y = np.random.binomial(size = n, n = 1, p = prob)
    
    # Modify the data according to censoring
    q1 = norm.ppf(cen[0])    # Compute the quantile values as the threshold
    q2 = cen[1]              # Since the nth percentile of U(0,1) is just n
    m1 = np.where(x1 < q1, 1, 0)    # Record down if the actual value is smaller than the threshold
    m2 = np.where(x2 < q2, 1, 0)    # 1 representing smaller (will be missing later) and 0 otherwise
    
    # Output the generated data with the extra missing indicator variable for subsequent data manipulation
    data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'm1': m1, 'm2': m2})
    return data


# Run all 4 cases (Different combinations of the censored rate of x1 and x2)
cen = [[0.25, 0.25], [0.25, 0.50], [0.50, 0.25], [0.50, 0.50]]
for i in range(0, 4):
    # Prepare the local variables
    reps = 50
    coef = pd.Series([0, 0, 0, 0, 0] , dtype = np.float64, index = ['Intercept', 'x1', 'x2', 'x3', 'x4'])
    mse = pd.Series([0, 0, 0, 0, 0] , dtype = np.float64, index = ['Intercept', 'x1', 'x2', 'x3', 'x4'])
    acc = 0
    knn = 0
    pval = 0

    # Specify the number of data entries, the censored rate and record the threshold
    n = 200
    cenrate = cen[i]
    q1 = norm.ppf(cenrate[0])   
    q2 = cenrate[1] 

    # We set the real minimum value of the variables subjected to detection limit
    minx1 = norm.ppf(0.01)
    minx2 = 0.01

    # GAIN imputation
    # Set seed for reproducibility, note that the GAIN part has to use the seed by TensorFlow
    np.random.seed(17)
    tf.set_random_seed(17) 

    # Define the GAIN hyperparameters
    gain_parameters = {
        'batch_size': 32,  
        'hint_rate': 0.9,
        'alpha': 100,
        'iterations': 10000
    }
    # The hyperparameters are get from hyperparameter tuning

    # Define the parameters for later use in normalization
    params = {
        'minx1': minx1,
        'minx2': minx2,
        'q1': q1,
        'q2': q2
    }

    # Perform simulation
    for j in range(0, reps):
        ### Step 1: Data Generation ###
        # Generate data of n entries with the specified censored rate
        data = datagen(n, cenrate)


        ### Step 2: Splitting Data ###
        # Split the data into train and test data using 80:20 ratio
        train_size = int(len(data) * 0.8)
        test_size = len(data) - train_size

        # Shuffle and split the DataFrame
        data = data.sample(frac = 1, random_state = 42).reset_index(drop = True)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]


        ### Step 3: Introduce Detection Limit to Training Data ###
        # Convert the values that are smaller than the threshold to NA
        train_data.loc[train_data['m1'] == 1, 'x1'] = np.nan    # If m1 = 1 (missing), then convert the original data to NaN
        train_data.loc[train_data['m2'] == 1, 'x2'] = np.nan   

        # Remove the columns m1 and m2; columns are along axis 1 in pandas
        train_data = train_data.drop(['m1', 'm2'], axis = 1)
        test_data = test_data.drop(['m1', 'm2'], axis = 1)

        # Convert Data Frame to numpy array
        train_data_np = train_data.to_numpy()  


        ### Step 4: Perform Logistic Regression and also Classification ###
        # Perform imputation using GAIN
        impute = gain(train_data_np, gain_parameters, params)

        # Convert numpy array back to Data Frame
        imputed_data = pd.DataFrame(impute, columns = train_data.columns)

        # Fit the data into logistic regression model with 5-fold cross-validation
        model = LogisticRegression()
        model.fit(imputed_data[['x1', 'x2', 'x3', 'x4']], imputed_data['y'])

        # Update the coefficients
        coef.iloc[0] += model.intercept_[0]   
        coef.iloc[1:] += model.coef_[0]   
        mse.iloc[0] += np.power(model.intercept_[0] - pd.Series([-3], index = ['Intercept']), 2)
        mse.iloc[1:] += np.power(model.coef_[0] - pd.Series([1, 1, 1, 1], index = ['x1', 'x2', 'x3', 'x4']), 2) 

        # Make predictions
        y_pred = model.predict(test_data.drop(['y'], axis = 1))

        # Measure the accuracy of each data point
        predict = np.where(y_pred == test_data['y'], 1, 0)

        # Evaluate the model
        acc += accuracy_score(test_data['y'], y_pred)

        # Fit the data into kNN with 5-fold cross-validation
        # Define the parameter grid
        param_grid = {'n_neighbors': list(range(1, 11))}  # Testing k values from 1 to 10

        # Initialize the k-NN model
        knn_model = KNeighborsClassifier()

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator = knn_model, param_grid = param_grid, cv = 5, scoring = 'accuracy')

        # Fit GridSearchCV to the data
        grid_search.fit(imputed_data[['x1', 'x2', 'x3', 'x4']], imputed_data['y'])

        # Train the best model on the entire training set
        best_knn_model = grid_search.best_estimator_

        # Make predictions using the best model
        knn_pred = best_knn_model.predict(test_data.drop(['y'], axis = 1))

        # Measure the accuracy of each data point
        knn_predict = np.where(knn_pred == test_data['y'], 1, 0)

        # Evaluate the model
        knn += accuracy_score(test_data['y'], knn_pred)

        # Perform McNemar's Test
        # Define class labels explicitly
        class_labels = [0, 1]  # Adjust based on your dataset's class labels

        # Compute confusion matrix
        cm = confusion_matrix(predict, knn_predict, labels = class_labels)

        # Perform McNemar test
        table = np.array([[cm[0, 0], cm[0, 1]], [cm[1, 0], cm[1, 1]]])
        result = mcnemar(table, correction = True)

        # Record the pvalue
        pval += result.pvalue


    ### Step 5: Summarise the results ###
    print()
    print('Case', i + 1 , ': The censored rate of x1 and x2 is', cenrate[0], 'and', cenrate[1], 'respectively.', end = '\n\n')

    # Compute the mean of estimated coefficients
    mean_coef = coef / reps
    print('The estimated coefficients is: ')
    print(mean_coef, end = '\n\n')

    # Compute the bias of the coefficients
    bias = mean_coef - pd.Series([-3, 1, 1, 1, 1], index = ['Intercept', 'x1', 'x2', 'x3', 'x4'])
    print('The estimated bias is: ')
    print(bias, end = '\n\n')

    # Compute the MSE of the coefficients
    mse = mse / reps
    print('The estimated MSE is: ')
    print(mse, end = '\n\n')

    # Compute the classification accuracy of logistic regression
    accuracy = acc / reps
    print('The logistic regression accuracy is: ', accuracy, end = '\n\n')

    # Compute the classification accuracy of kNN
    knn_accuracy = knn / reps
    print('The kNN accuracy is: ', knn_accuracy, end = '\n\n')

    # Compute the p-value of the McNemar's Test
    avg_pval = pval / reps
    print('The p-value is: ', avg_pval, end = '\n\n')