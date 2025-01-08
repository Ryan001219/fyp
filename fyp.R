# Import necessary libraries
library(dplyr)
library(caret)
library(mice)
library(missForest)
library(e1071)

## Logistic regression model: logit(p) = b0 + b1x1 + b2x2 + b3x3 + b4x4
## Note that we have x1 and x2 subjected to detection limit

## Assumptions ##
# 1. We assume that we know which entries are subjected to detection limits
# 2. We assume that we know the real minimum value of the variables subjected to detection limit


##### Simulation #####

## Threshold for censoring of x1 and x2: lower quartile or median of the distribution
## i.e. Data smaller than threshold will be taking the threshold value

### Create simulated datasets as follows ###
# By referencing the simulation technique mentioned in Chiou et al.

datagen = function(n, cen) {
  # Create random data according to the distributions
  x1 = rnorm(n, mean = 0, sd = 1)    ## x1 ~ Normal(0,1)
  x2 = runif(n, min = 0, max = 1)    ## x2 ~ Uniform(0,1)
  x3 = rnorm(n, mean = 0, sd = 1)    ## x3 ~ Normal(0,1)
  x4 = rbinom(n, size = 1, prob = 0.5)    ## x4 ~ Bernoulli(0.5) = Binomial(1, 0.5)
  
  # We set the true coefficients to be b0 = b1 = b2 = b3 = b4 = 1
  # We generate the response variable y as follows
  logit = x1 + x2 + x3 + x4 + 1
  prob = 1 / (1 + exp(-logit))    
  y = rbinom(n, 1, prob)
  
  # Modify the data according to censoring
  q1 = qnorm(cen[1])    # Compute the quantile values as the threshold
  q2 = qunif(cen[2])
  m1 = ifelse(x1 < q1, 1, 0)    # Record down if the actual value is smaller than the threshold
  m2 = ifelse(x2 < q2, 1, 0)    # 1 representing smaller (will be missing later) and 0 otherwise
  
  # Output the generated dataset with the extra missing indicator variable for subsequent data manipulation
  data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4, m1 = m1, m2 = m2)
}


## We create a function to change a certain value from the original range to the new range
convert = function(x, oldmin, oldmax, newmin, newmax) {
  # Normalize the values 
  normalized = (x - oldmin) / (oldmax - oldmin)
  
  # Rescale the value to the new range
  rescaled = normalized * (newmax - newmin) + newmin
  
  return(rescaled)
}


## We create functions to reduce redundancy and improve the readability of the code
# Function for logistic regression model fitting with 5-fold cross-validation
logreg_fit = function(data) {
  model = train(y ~ ., data, method = 'glm', family = binomial, 
                trControl = trainControl(method = 'cv', number = 5))
  return(model)
}

# Function for evaluating the different metrics of logistic regression model
logreg_stat = function(model) {
  # Compute bias and MSE
  bias = coef(model$finalModel) - rep(1,5)
  mse = (coef(model$finalModel) - rep(1,5))^2
  
  # Test the classification accuracy using test data and measure the accuracy of each data point
  pred = predict(model, newdata = test.data) 
  predict = ifelse(pred == test.data$y, 1, 0)
  acc = mean(predict)  
  
  return(list(bias = bias, mse = mse, predict = predict, acc = acc))
}

# Function for kNN fitting with 5-fold cross-validation
knn_fit = function(data) {
  model = train(y ~ ., data, method = 'knn', 
                preProcess = c("center", "scale"), 
                tuneLength = 10,  # Try 10 different values of k
                trControl = trainControl(method = 'cv', number = 5))
  return(model)
}

# Function for evaluating the different metrics of kNN
knn_stat = function(model) {
  # Test the classification accuracy using test data and measure the accuracy of each data point
  pred = predict(model, newdata = test.data) 
  predict = ifelse(pred == test.data$y, 1, 0)
  acc = mean(predict) 
  
  return(list(predict = predict, acc = acc))
}

# Function for carrying out McNemar's Test
mcnemar = function(logreg, knn) {
  # Make sure the resulting table is a square table (consists of both 0 and 1 classes)
  logreg = factor(logreg, levels = c(0,1))
  knn = factor(knn, levels = c(0,1))
  
  # Perfect agreement of the two vectors will automatically have a p-value of 1
  if (all(logreg == knn)) {
    return(1)
  } else{
    # Create a table that represents the number where model A and B predicted correctly or incorrectly
    tab = table(logreg, knn)
    
    # Perform McNemar's Test to compare the predictive accuracy of two models and record the p-value
    result = mcnemar.test(tab, correct = T)
    
    return(result$p.value)
  }
}


### Simulation Part ###

## Run all 4 cases (Different combinations of the censored rate of x1 and x2)
cen = list(c(0.25, 0.25), c(0.25, 0.50), c(0.50, 0.25), c(0.50, 0.50))

# We set the real minimum value of the variables subjected to detection limit
minx1 = qnorm(0.01)
minx2 = qunif(0.01)

# Prepare the local variables
reps = 1000
count = 0
bias.cc = bias.mice = bias.miss = bias.svs = rep(0, 5)
mse.cc = mse.mice = mse.miss = mse.svs = rep(0, 5)
acc.cc = acc.mice = acc.miss = acc.svs = 0
knn.cc = knn.mice = knn.miss = knn.svs = 0
pval.cc = pval.mice = pval.miss = pval.svs = 0

set.seed(17)

## Specify the number of data entries, the censored rate and record the threshold
## Change the index of argument cen to simulate different censored rate ##
n = 200
cenrate = cen[[1]]     # Change this value to get cases 1 to 4
q1 = qnorm(cenrate[1])    
q2 = qunif(cenrate[2]) 


# For loop for simulation
for (i in 1:reps){   
  ### Step 1: Data Generation ###
  # Generate data and convert the response variable y into 2 levels
  data = datagen(n, cenrate)
  data$y = as.factor(data$y)
  
  
  ### Step 2: Splitting Data ###
  # Split the data into train and test data using 80:20 ratio
  training.idx = sample(1: nrow(data), size = nrow(data)*0.8) 
  train.data = data[training.idx, ] 
  test.data = data[-training.idx, ]
  
  
  ### Step 3: Introduce Detection Limit to Training Data ###
  # Convert the values that are smaller than the threshold to NA
  cen.data = train.data %>% mutate(x1 = ifelse(m1 == 1, NA, x1), x2 = ifelse(m2 == 1, NA, x2))
  
  # Prepare the data by discarding the columns m1 and m2
  data.cc = data.mice = data.miss = data.svs = cen.data %>% select(y, x1, x2, x3, x4)
  test.data = test.data %>% select(y, x1, x2, x3, x4)
  
  
  ### Step 4: Perform Logistic Regression, kNN and Classification ###
  
  ## Method 1: Complete Case Analysis ##
  # Prepare the training data by removing entries that having missing data
  data.cc = data.cc[complete.cases(data.cc),] 
  
  # If y contains both 0s and 1s, we will record the variable 'y' to be TRUE, same applies to x4
  y = ifelse(sum(data.cc$y == 0) > 0 && sum(data.cc$y == 1) > 0, TRUE, FALSE)
  x4 = ifelse(sum(data.cc$x4 == 0) > 0 && sum(data.cc$x4 == 1) > 0, TRUE, FALSE)
  
  # Fit the data into logistic regression model while checking convergence of the model
  if (y == TRUE && x4 == TRUE){
    model.cc = tryCatch({
      logreg_fit(data.cc)
    } , warning = function(w){
      if (grepl("algorithm did not converge", conditionMessage(w))) {
        return(NULL)  # Return NULL to indicate non-convergence
      }
    })
  } else {
    # There are missing classes in the binary variables and will print error message during model fitting
    model.cc = NULL
  }
  
  # Record the necessary data if the model is valid
  if (!is.null(model.cc) && y == TRUE && x4 == TRUE){
    result.cc = logreg_stat(model.cc)
    
    bias.cc = bias.cc + result.cc$bias
    mse.cc = mse.cc + result.cc$mse
    acc.cc = acc.cc + result.cc$acc
    
    # Train kNN according to this training set and test set
    knn_mod.cc = knn_fit(data.cc) 
    knn_result.cc = knn_stat(knn_mod.cc)
    
    knn.cc = knn.cc + knn_result.cc$acc
    
    # Perform McNemar Test
    pval.cc = pval.cc + mcnemar(result.cc$predict, knn_result.cc$predict)
    
  } else {
    # Track how many invalid cases (Model diverges or one of the classes is missing)
    count = count + 1
  }
  
  
  ## Method 2: MICE Imputation with Imputation Restriction ##
  # Prepare the variables for pooling results of multiple imputation
  bias = mse = acc = knn = auclr = aucknn = pval = 0
  
  # Perform the MICE imputation
  data.mice = mice(data.mice, m = 5, method = "norm", printFlag = FALSE)
  
  # Loop through each imputed dataset and process it
  for (p in 1:5) {
    imputed = complete(data.mice, p)
    
    # Restrict the imputed value to be smaller than the threshold
    imputed$x1[cen.data$m1 == 1 & imputed$x1 > q1] = sapply(imputed$x1[cen.data$m1 == 1 & imputed$x1 > q1], function(x) convert(x, minx1, max(imputed$x1), minx1, q1))
    imputed$x2[cen.data$m2 == 1 & imputed$x2 > q2] = sapply(imputed$x2[cen.data$m2 == 1 & imputed$x2 > q2], function(x) convert(x, minx2, max(imputed$x2), minx2, q2))
    
    # Fit the data into logistic regression model
    model.mice = logreg_fit(imputed)
    
    # Record the necessary data 
    result.mice = logreg_stat(model.mice)
    
    bias = bias + result.mice$bias
    mse = mse + result.mice$mse
    acc = acc + result.mice$acc
    
    # Train kNN according to this training set and test set
    knn_mod.mice = knn_fit(imputed)
    knn_result.mice = knn_stat(knn_mod.mice)
    
    knn = knn + knn_result.mice$acc
    
    # Perform McNemar Test
    pval = pval + mcnemar(result.mice$predict, knn_result.mice$predict)
  }
  
  # Now add the pooled result
  bias.mice = bias.mice + bias/5
  mse.mice = mse.mice + mse/5
  acc.mice = acc.mice + acc/5
  knn.mice = knn.mice + knn/5
  pval.mice = pval.mice + pval/5
  
  ## Method 3: missForest Imputation with Imputation Restriction ##
  # Perform missForest imputation and retrieve the imputed data
  data.miss = missForest(data.miss)$ximp
  
  # Restrict the imputed value to be smaller than the threshold
  data.miss$x1[cen.data$m1 == 1 & data.miss$x1 > q1] = sapply(data.miss$x1[cen.data$m1 == 1 & data.miss$x1 > q1], function(x) convert(x, minx1, max(data.miss$x1), minx1, q1))
  data.miss$x2[cen.data$m2 == 1 & data.miss$x2 > q2] = sapply(data.miss$x2[cen.data$m2 == 1 & data.miss$x2 > q2], function(x) convert(x, minx2, max(data.miss$x2), minx2, q2))
  
  # Fit the data into logistic regression model
  model.miss = logreg_fit(data.miss)
  
  # Record the necessary data 
  result.miss = logreg_stat(model.miss)
  
  bias.miss = bias.miss + result.miss$bias
  mse.miss = mse.miss + result.miss$mse
  acc.miss = acc.miss + result.miss$acc
  
  # Train kNN according to this training set and test set
  knn_mod.miss = knn_fit(data.miss)
  knn_result.miss = knn_stat(knn_mod.miss)
  
  knn.miss = knn.miss + knn_result.miss$acc
  
  # Perform McNemar Test
  pval.miss = pval.miss + mcnemar(result.miss$predict, knn_result.miss$predict)
  
  
  ## Method 4: Substitute the missing value with a certain value ##
  v1 = (minx1 + q1)/2
  v2 = (minx2 + q2)/2
  data.svs$x1[cen.data$m1 == 1] = v1
  data.svs$x2[cen.data$m2 == 1] = v2
  
  # Fit the data into logistic regression model
  model.svs = logreg_fit(data.svs)
  
  # Record the necessary data 
  result.svs = logreg_stat(model.svs)
  
  bias.svs = bias.svs + result.svs$bias
  mse.svs = mse.svs + result.svs$mse
  acc.svs = acc.svs + result.svs$acc
  
  # Train kNN according to this training set and test set
  knn_mod.svs = knn_fit(data.svs)
  knn_result.svs = knn_stat(knn_mod.svs)
  
  knn.svs = knn.svs + knn_result.svs$acc
  
  # Perform McNemar Test
  pval.svs = pval.svs + mcnemar(result.svs$predict, knn_result.svs$predict)
}

# Regenerate data so that the total number of replications are the same
j = 0     # Initiate the running count to be 0

while (j < count){
  data = datagen(n, cenrate)
  
  data$y = as.factor(data$y)
  
  training.idx = sample(1: nrow(data), size = nrow(data)*0.8) 
  train.data = data[training.idx, ] 
  test.data = data[-training.idx, ]
  
  cen.data = train.data %>% mutate(x1 = ifelse(m1 == 1, NA, x1), x2 = ifelse(m2 == 1, NA, x2))
  
  data.cc = cen.data %>% select(y, x1, x2, x3, x4)
  test.data = test.data %>% select(y, x1, x2, x3, x4)
  
  data.cc = data.cc[complete.cases(data.cc),] 
  
  y = ifelse(sum(data.cc$y == 0) > 0 && sum(data.cc$y == 1) > 0, TRUE, FALSE)
  x4 = ifelse(sum(data.cc$x4 == 0) > 0 && sum(data.cc$x4 == 1) > 0, TRUE, FALSE)
  
  if (y == TRUE && x4 == TRUE){
    model.cc = tryCatch({
      logreg_fit(data.cc)
    } , warning = function(w){
      if (grepl("algorithm did not converge", conditionMessage(w))) {return(NULL)}
    })
  } else {model.cc = NULL}
  
  if (!is.null(model.cc) && y == TRUE && x4 == TRUE){
    result.cc = logreg_stat(model.cc)
    
    bias.cc = bias.cc + result.cc$bias
    mse.cc = mse.cc + result.cc$mse
    acc.cc = acc.cc + result.cc$acc
    
    knn_mod.cc = knn_fit(data.cc)
    knn_result.cc = knn_stat(knn_mod.cc)
    
    knn.cc = knn.cc + knn_result.cc$acc
    
    pval.cc = pval.cc + mcnemar(result.cc$predict, knn_result.cc$predict)
    
    # Update the running count if the model is valid
    j = j + 1
  } 
}


### Step 5: Summarise the results ###
# Compute the bias of the coefficients
bias.cc = bias.cc / reps
bias.mice = bias.mice / reps
bias.miss = bias.miss / reps
bias.svs = bias.svs / reps

# Compute the MSEs of the coefficients
mse.cc = mse.cc / reps
mse.mice = mse.mice / reps
mse.miss = mse.miss / reps
mse.svs = mse.svs / reps

# Check the classification accuracy of logistic regression
acc.cc = acc.cc / reps
acc.mice = acc.mice / reps
acc.miss = acc.miss / reps
acc.svs = acc.svs / reps

# Check the classification accuracy of kNN
knn.cc = knn.cc / reps
knn.mice = knn.mice / reps
knn.miss = knn.miss / reps
knn.svs = knn.svs / reps

# Check the p-value regarding the McNemar Test
pval.cc = pval.cc / reps
pval.mice = pval.mice / reps
pval.miss = pval.miss / reps
pval.svs = pval.svs / reps

# Summary: Print the biases, MSEs and the accuracy in a data frame for comparison 
bias = rbind(bias.cc, bias.svs, bias.mice, bias.miss)
mse = rbind(mse.cc, mse.svs, mse.mice, mse.miss)
acc = rbind(acc.cc, acc.svs, acc.mice, acc.miss)
knn = rbind(knn.cc, knn.svs, knn.mice, knn.miss)
pval = rbind(pval.cc, pval.svs, pval.mice, pval.miss)