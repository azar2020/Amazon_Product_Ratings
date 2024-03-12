# Predicting Amazon product rating

# Importing R libraries to be used for analysis
library(readxl)
require(glmnet)

# Remove all objects from the global environment
rm(list = ls())

# Set working directory
setwd("C:/Users/aazar/OneDrive/Desktop/Computational stat/data")

# Read data from Excel file
mydata = read_excel("Amazon_dataset.xlsx")
head(mydata)

# Plotting
plot(mydata$discount_percentage, mydata$rating, xlab = "discount_percentage", ylab = "rating")
plot(mydata$actual_price, mydata$rating, xlab = "actual_price", ylab = "rating")
plot(mydata$rating_count, mydata$rating, xlab = "rating_count", ylab = "rating")
boxplot(mydata$rating, main = "Box plot of data with small standard deviation", ylab = "rating")

# LASSO Without Cross_Validation
n = dim(mydata)[1]

# 75% training, 25% testing
n_training = 1075
n_testing = n - n_training

# Set seed and generate training index
set.seed(6)
training_indices = sample(c(1:n), size = n_training)

# Create training and testing datasets
training_data = mydata[training_indices,]
testing_data = mydata[-training_indices,]

# Use all inputs
trainingX = model.matrix(rating ~ ., data = training_data)
testingX = model.matrix(rating ~ ., data = testing_data)

trainingy = training_data$rating
testingy = testing_data$rating

p = dim(trainingX)[2] - 1

# Find the optimal lambda without cross-validation
lambda_values = seq(from = 0, to = 0.1, by = 0.0001)
n_lambda_values = length(lambda_values)
testing_loss = matrix(0, nrow = n_lambda_values)
testing_loss_mae = matrix(0, nrow = n_lambda_values)


for(i in 1:n_lambda_values){
  lambda = lambda_values[i]
  
  # Recompute glmnet model and yHat
  model = glmnet(trainingX, trainingy, family = 'gaussian', alpha = 1, lambda = lambda)
  yHat = predict(model, newx = testingX)
  
  # Compute and store testing loss
  testing_loss[i] = mean((yHat - testingy)^2)
  testing_loss_mae[i] = mean(abs(yHat - testingy))
}

# Plot the testing loss against lambda values
plot(lambda_values, testing_loss, xlab = 'lambda', ylab = 'MSE_LASSO')

# Find minimum loss and corresponding lambda
min_loss = min(testing_loss)
min_loss_lambda = lambda_values[which.min(testing_loss)]

# Computing MSE and MAE
MSE_LASSO_without_CV = min_loss
MAE_LASSO_without_CV = min(testing_loss_mae)

# Fit the LASSO model with the optimal lambda
lambda = min_loss_lambda
model = glmnet(trainingX, trainingy, family = 'gaussian', alpha = 1, lambda = lambda)
beta_LASSO = model$beta

# LASSO WITH CROSS VALIDATION

# Use K = 5
K = 5
n_per_fold = 282

# Set seed and create folds
set.seed(0)
folds = list()
shuffled_index = sample(c(1:n))
for(k in c(1:(K - 1))){
  folds[[k]] = shuffled_index[c((1 + (k - 1) * n_per_fold):(k * n_per_fold))]
}  
folds[[K]] = shuffled_index[c((1 + (K - 1) * n_per_fold):n)]

# For a sequence of lambda values 
lambda_values = seq(from = 0, to = 0.1, by = 0.001)
n_lambda_values = length(lambda_values)

# Initialize MSE and MAE values for the ridge regression
MSE_LASSO_with_CV = matrix(0, nrow = n_lambda_values, ncol = K)
MAE_LASSO_with_CV = matrix(0, nrow = n_lambda_values, ncol = K)

for(k in c(1:K)){
  training_data = mydata[-folds[[k]],]
  n_training = dim(training_data)[1]
  testing_data = mydata[folds[[k]],]
  n_testing = dim(testing_data)[1]
  
  # Create input matrix and output vector for training and testing data
  trainingX = model.matrix(rating ~ ., data = training_data)
  trainingX0 = trainingX[, -1] 
  trainingY = training_data$rating
  
  testingX = model.matrix(rating ~ ., data = testing_data)
  testingX0 = testingX[, -1]
  testingY = testing_data$rating
  
  # Scaling training and testing inputs
  x_sd = apply(trainingX0, 2, sd)
  x_mean = apply(trainingX0, 2, mean)
  
  trainingX0 = t((t(trainingX0) - x_mean) / x_sd)
  testingX0 = t((t(testingX0) - x_mean) / x_sd)
  
  # Create testingX (with intercept) for predictions and MSE computation
  testingX = cbind(matrix(1, nrow = n_testing), testingX0)
  
  # Perform LASSO for a range of lambda values
  for(i in c(1:n_lambda_values)){ 
    lambda = lambda_values[i]
    fit = glmnet(trainingX0, trainingY - mean(trainingY), alpha = 1, lambda = lambda, intercept = FALSE)
    bHat = matrix(coef(fit), nrow = (p + 1))
    bHat[1] = mean(trainingY)
    testingYhat = testingX %*% bHat
    MSE_LASSO_with_CV[i, k] = sum((testingYhat - testingY)^2) / n_testing
    MAE_LASSO_with_CV[i, k] = sum(abs(testingYhat - testingY)) / n_testing
  }
}

# Take average MSE and MAE across folds
MSE_LASSO_with_CV = apply(MSE_LASSO_with_CV, 1, mean)
MAE_LASSO_with_CV = apply(MAE_LASSO_with_CV, 1, mean)

# Plot MSE values as a function of lambda
plot(lambda_values, MSE_LASSO_with_CV, xlab = "lambda", ylab = 'MSE_LASSO', main = 'LASSO Regression - MSE of testing data', bty = 'n')
plot(lambda_values, MSE_LASSO_with_CV, xlab = 'lambda', ylab = 'MSE_LASSO')

# Find minimum MSE and corresponding lambda
min_MSE_LASSO_with_CV = min(MSE_LASSO_with_CV)
min_MSE_lambda_LASSO_with_CV = lambda_values[which.min(MSE_LASSO_with_CV)]

# Fit a final model with the chosen lambda
X = model.matrix(rating ~ ., data = mydata)
Y = mydata$rating
fit = glmnet(X, Y, alpha = 1, lambda = 0.013)
fit
fit$a0
fit$beta

# Kernel-Smoothing

# Gaussian Kernel
t = seq(from = -2, to = 2, by = 0.01)
gaussian_kernel = function(t){
  return((2 * pi)^(-1/2) * exp(-t^2/2))
}
y = gaussian_kernel(t)

# Plot Gaussian Kernel
plot(t, y, type = 'l', lwd = 2, xlab = 't', ylab = 'D(t)', main = 'Gaussian Kernel', bty = 'n', ylim = c(0, 1), xlim = c(-2, 2))

# Write a Kernel smoothing function
multivariate_kernel_smoothing = function(x0, X, Y, K, lambda = 1){
  distance = function(a, b){
    return(sqrt(sum((a - b)^2)))
  }
  N = dim(X)[1]
  w = matrix(0, nrow = N)
  for(i in c(1:N)){
    w[i] = K(distance(x0, X[i,]) / lambda)
  }
  return(sum(w * Y) / sum(w))
}

k = 5
n_training = (k - 1) * n_per_fold
n_testing = n_per_fold

# Set seed and create folds
set.seed(0)
folds = list()
shuffled_index = sample(c(1:n))
for(fold in c(1:k)){
  folds[[fold]] = shuffled_index[c((1 + (fold - 1) * n_per_fold):(fold * n_per_fold))]
}  

# For a sequence of lambda values 
lambda_values = seq(from = 0.9, to = 1.8, by = 0.1)
n_lambda_values = length(lambda_values)

# Initialize MSE values for the ridge regression
MSE_kernel_smoothing = matrix(0, nrow = n_lambda_values, ncol = k)
MAE_kernel_smoothing = matrix(0, nrow = n_lambda_values, ncol = k)

for(fold in c(1:k)){
  training_data = mydata[-folds[[fold]],]
  testing_data = mydata[folds[[fold]],]
  
  # Create input matrix and output vector for training and testing data
  trainingX = model.matrix(rating ~ 0 + ., data = training_data)
  trainingY = matrix(training_data$rating, nrow = n_training)
  
  testingX = model.matrix(rating ~ 0 + ., data = testing_data)
  testingY = matrix(testing_data$rating, nrow = n_testing)
  
  # Scaling training and testing inputs
  x_sd = apply(trainingX, 2, sd)
  x_mean = apply(trainingX, 2, mean)
  
  trainingX = t((t(trainingX) - x_mean) / x_sd)
  testingX = t((t(testingX) - x_mean) / x_sd)
  
  # Perform ridge regression for a range of lambda values
  for(i in c(1:n_lambda_values)){ 
    lambda = lambda_values[i]
    testingYhat = matrix(0, nrow = n_testing)
    for(j in c(1:n_testing)){
      testingYhat[j] = multivariate_kernel_smoothing(testingX[j,], trainingX, trainingY, gaussian_kernel, lambda = lambda)
    }
    MAE_kernel_smoothing[i, fold] = sum(abs(testingYhat - testingY)) / n_testing
    MSE_kernel_smoothing[i, fold] = sum((testingYhat - testingY)^2) / n_testing
  }
}

# Take average MSE and MAE across folds
MSE_kernel_smoothing = apply(MSE_kernel_smoothing, 1, mean)
MAE_kernel_smoothing = apply(MAE_kernel_smoothing, 1, mean)

# Plot MSE values as a function of lambda
plot(lambda_values, MSE_kernel_smoothing, xlab = 'lambda', ylab = 'MSE_kernel_smoothing', main = 'MSE from cross validation')

# Find minimum MSE and corresponding lambda for kernel smoothing
min_MSE_kernel_smoothing = min(MSE_kernel_smoothing)
min_MSE_lambda_kernel_smoothing = lambda_values[which.min(MSE_kernel_smoothing)]

min_MSE_kernel_smoothing
MAE_kernel_smoothing
