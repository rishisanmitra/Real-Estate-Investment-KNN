# Rishi (Author)
# Load the necessary libraries 
library(caret) # used for data splitting
library(dplyr) # used for data manipulation 
library(fastDummies) # used for dummy variable creation
library(class)
library(e1071)

# Load the data
housing.df <- read.csv("WestRoxburyCat.csv")

# Explore Data
summary(housing.df)

# Removing the total value and tax from the data
housing.df <- housing.df %>% select(-TOTAL.VALUE, -TAX)

# Separate the target variable and predictors
X <- housing.df %>% select(-CAT.VALUE)
Y <- as.factor(housing.df$CAT.VALUE)

# Convert the Categorical Variable (REMODEL) into factor
X$REMODEL <- as.factor(X$REMODEL)

# Create Dummy Columns for each level in REMODEL
X <- dummy_cols(X, 
                select_columns = "REMODEL",
                remove_selected_columns = TRUE)

# Re-scale the predictors data 
X_scaled <- scale(X)

# Create data partitions for training(70%) and testing(30%) data
set.seed(1998)
train_index <- createDataPartition(Y, p = 0.7, list=FALSE)
X_Train <- X_scaled[train_index, ]
X_Test <- X_scaled[-train_index, ]
Y_Train <- Y[train_index]
Y_Test <- Y[-train_index]
str(Y_Train)
# Percentage of houses in the original data >= $500K
percentage_OG <- sum(housing.df$CAT.VALUE == 1)/nrow(housing.df) * 100

# Display Percentage
percentage_OG

# Percentage of houses in the training data >= $500K
percentage_Train <- sum(Y_Train == 1)/length(Y_Train) * 100

# Display Percentage
percentage_Train

# Percentage of houses in the testing data >= $500K
percentage_Test <- sum(Y_Test == 1)/length(Y_Test) * 100

# Display Percentage
percentage_Test

# KNN initially with K = 10 
k_initial <- 10
Y_Pred <- knn(train = X_Train, test = X_Test, cl = Y_Train, k = k_initial)

# Evaluating Model Performance
conf_matrix <- confusionMatrix(Y_Pred, Y_Test, positive = "1")
conf_matrix

# Tuning the K value using tune.knn() function
set.seed(1997)
tune_result <- tune.knn(x = X_Train, y = Y_Train, k = 1:30)
tune_result
plot(tune_result)

# KNN using best K
best_k <- tune_result$best.parameters$k
Y_Pred_best <- knn(train = X_Train, test = X_Test, cl = Y_Train, k = best_k)

# Evaluate performance
conf_matrix_best <- confusionMatrix(Y_Pred_best, Y_Test, positive = "1")
conf_matrix_best

#--------------------------------------------------------------------------------
# Determining best K using sensitivity instead of accuracy.
k_values <- 1:20
sensitivity_values <- numeric(length(k_values))

for(i in seq_along(k_values)) {
  k <- k_values[i]

  # Predict on training data
  Y_pred_train <- knn(train = X_Train, test = X_Test, cl = Y_Train, k = k)
  
  # Confusion matrix
  cm <- confusionMatrix(Y_pred_train, Y_Test, positive = "1")
  # Store sensitivity (recall for positive class)
  sensitivity_values[i] <- cm$byClass["Sensitivity"]
}
#
sensitivity_values
#Find k with maximum sensitivity
best_k_sens <- k_values[which.max(sensitivity_values)]
best_k_sens
#
Y_pred_train <- knn(train = X_Train, test = X_Test, cl = Y_Train, k = 3)
#
## Confusion matrix
cm <- confusionMatrix(Y_pred_train, Y_Test, positive = "1")
cm
