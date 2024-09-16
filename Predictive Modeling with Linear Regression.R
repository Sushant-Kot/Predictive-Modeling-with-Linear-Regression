# Install and load necessary packages
library(tidyverse)
library(caret)
library(broom)
library(glmnet)
library(pROC)

# Load and preprocess the dataset
data(mtcars)
df <- mtcars

# Feature Engineering: Create interaction terms
df <- df %>%
  mutate(HP_WT_Interaction = hp * wt)

# Data preprocessing: Standardize the features
df_scaled <- df %>%
  select(mpg, everything()) %>%
  scale() %>%
  as.data.frame()  # Convert back to data frame

# Preserve the original column names (scale() removes them)
colnames(df_scaled) <- colnames(df)

# Split the data into training and test sets
set.seed(123) # for reproducibility
trainIndex <- createDataPartition(df_scaled$mpg, p = 0.8, list = FALSE)
trainData <- df_scaled[trainIndex, ]
testData <- df_scaled[-trainIndex, ]

# Define features and target
x_train <- as.matrix(trainData %>% select(-mpg))
y_train <- trainData$mpg
x_test <- as.matrix(testData %>% select(-mpg))
y_test <- testData$mpg

# Model Building: Fit a linear regression model
lm_model <- lm(mpg ~ ., data = trainData)

# Model Tuning: Ridge and Lasso Regression
alpha_values <- c(0, 1) # Ridge (alpha=0) and Lasso (alpha=1)
cv_results <- lapply(alpha_values, function(a) {
  cv.glmnet(x_train, y_train, alpha = a)
})

# Best Lambda values
best_lambda_ridge <- cv_results[[1]]$lambda.min
best_lambda_lasso <- cv_results[[2]]$lambda.min

# Predictions
lm_predictions <- predict(lm_model, newdata = testData)
ridge_predictions <- predict(cv_results[[1]], s = best_lambda_ridge, newx = x_test)
lasso_predictions <- predict(cv_results[[2]], s = best_lambda_lasso, newx = x_test)

# Evaluation
lm_performance <- postResample(lm_predictions, y_test)
ridge_performance <- postResample(ridge_predictions, y_test)
lasso_performance <- postResample(lasso_predictions, y_test)

print(lm_performance)
print(ridge_performance)
print(lasso_performance)

# Visualization of actual vs predicted values for Lasso model
plot_df <- data.frame(
  Actual = y_test,
  Lasso_Predicted = as.numeric(lasso_predictions)  # Ensure predictions are numeric
)

ggplot(plot_df, aes(x = Actual, y = Lasso_Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Actual vs Predicted MPG (Lasso Regression)", x = "Actual MPG", y = "Predicted MPG") +
  theme_minimal()

# Save the plot to a file
ggsave("lasso_actual_vs_predicted.png")
