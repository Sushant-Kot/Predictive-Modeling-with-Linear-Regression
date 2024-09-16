# Install and load necessary packages
if (!require("tidyverse")) install.packages("tidyverse", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("broom")) install.packages("broom", dependencies = TRUE)

library(tidyverse)
library(caret)
library(broom)

# Load the dataset
data(mtcars)
df <- mtcars

# Display the first few rows of the dataset
head(df)

# Data preprocessing (check for missing values)
sum(is.na(df))

# Define the target variable and features
target <- "mpg"
features <- setdiff(names(df), target)

# Split the data into training and test sets
set.seed(123) # for reproducibility
trainIndex <- createDataPartition(df[[target]], p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

# Build the linear regression model
model <- lm(mpg ~ ., data = trainData)

# Summarize the model
summary(model)

# Make predictions on the test set
predictions <- predict(model, newdata = testData)

# Evaluate the model performance
performance <- postResample(predictions, testData[[target]])
print(performance)

# Plot actual vs predicted values
plot_df <- data.frame(
  Actual = testData[[target]],
  Predicted = predictions
)

ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Actual vs Predicted MPG", x = "Actual MPG", y = "Predicted MPG")

# Save the plot to a file
ggsave("actual_vs_predicted.png")
