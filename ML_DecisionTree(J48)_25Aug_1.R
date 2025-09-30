# Decision Tree (J48)
# Load Libraries
library(RWeka)
library(caret)
library(lubridate)
library(dplyr)

# Load dataset
df <- read.csv("../Dataset/Test/Netflix Userbase.csv")

df$Last_Payment_Date <- as.Date(df$Last_Payment_Date, format = "%m/%d/%Y")
df$Days_Since_Last_Payment <- as.numeric(Sys.Date() - df$Last_Payment_Date)

df$Join_Date <- as.Date(df$Join_Date, format = "%m/%d/%Y")
df$Join_month <- month(df$Join_Date)
df$Days_Since_Join_date <- as.numeric(Sys.Date() - df$Join_Date)

df <- df %>% select(-User_ID, -Plan_Duration, -Join_Date, -Last_Payment_Date)
df <- df[ , c(setdiff(names(df), "Subscription_Type"), "Subscription_Type")]

# Convert relevant columns to factors
df$Country <- as.factor(df$Country)
df$Gender <- as.factor(df$Gender)
df$Device <- as.factor(df$Device)
df$Subscription_Type <- as.factor(df$Subscription_Type)
#df$Subscription_Type <- factor(df$Subscription_Type, levels=c("Premium","Standard","Basic"), ordered = TRUE)

#View(df)


# Train-test split
set.seed(123)
trainIndex <- createDataPartition(df$Subscription_Type, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

# Define tuning grid
grid <- expand.grid(.C = c(0.01, 0.001, 0.0001),
                    .M = c(3, 5, 10, 20, 30))

# Train with cross-validation
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)

# + Days_Since_Join_date
# + Days_Since_Last_Payment
# + Monthly_Revenue

j48_tuned <- train(Subscription_Type ~ Join_month + Country + Device + Age + Gender 
                   + Days_Since_Join_date + Days_Since_Last_Payment + Monthly_Revenue,
                   data = trainData,
                   method = "J48",
                   trControl = ctrl,
                   tuneGrid = grid)

# Best tuning parameters
print(j48_tuned$bestTune)

# Predict using the best model
pred <- predict(j48_tuned, newdata = testData)
confusionMatrix(pred, testData$Subscription_Type)

#2 Evaluation Metrics 
# Load required libraries
library(yardstick)
library(ggplot2)
library(gridExtra)

# Make predictions on training and test sets
train_pred <- predict(j48_tuned, newdata = trainData)
test_pred <- predict(j48_tuned, newdata = testData)

# Function to calculate metrics
calculate_metrics <- function(predictions, actual) {
  # Create confusion matrix
  cm <- table(Predicted = predictions, Actual = actual)
  
  # Calculate metrics for each class
  classes <- levels(actual)
  metrics_list <- list()
  
  for (class in classes) {
    TP <- cm[class, class]
    FP <- sum(cm[class, ]) - TP
    FN <- sum(cm[, class]) - TP
    TN <- sum(cm) - TP - FP - FN
    
    accuracy <- (TP + TN) / sum(cm)
    precision <- TP / (TP + FP)
    recall <- TP / (TP + FN)
    f1_score <- 2 * (precision * recall) / (precision + recall)
    
    metrics_list[[class]] <- data.frame(
      Class = class,
      Accuracy = accuracy,
      Precision = precision,
      Recall = recall,
      F1_Score = f1_score
    )
  }
  
  # Calculate overall metrics
  overall_accuracy <- sum(diag(cm)) / sum(cm)
  
  # Macro-averaged metrics
  macro_precision <- mean(sapply(metrics_list, function(x) x$Precision))
  macro_recall <- mean(sapply(metrics_list, function(x) x$Recall))
  macro_f1 <- mean(sapply(metrics_list, function(x) x$F1_Score))
  
  overall_metrics <- data.frame(
    Class = "Overall",
    Accuracy = overall_accuracy,
    Precision = macro_precision,
    Recall = macro_recall,
    F1_Score = macro_f1
  )
  
  # Combine all metrics
  all_metrics <- do.call(rbind, metrics_list)
  all_metrics <- rbind(all_metrics, overall_metrics)
  
  return(list(
    confusion_matrix = cm,
    metrics = all_metrics
  ))
}

# Function to plot colored confusion matrix
plot_colored_confusion_matrix <- function(predictions, actual, title) {
  cm <- table(Predicted = predictions, Actual = actual)
  cm_df <- as.data.frame(cm)
  
  ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "white", size = 6) +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Calculate metrics for training set
train_results <- calculate_metrics(train_pred, trainData$Subscription_Type)

# Calculate metrics for test set
test_results <- calculate_metrics(test_pred, testData$Subscription_Type)

# Print results for training set
cat("=== TRAINING SET RESULTS ===\n")
print(train_results$confusion_matrix)

cat("\n=== TRAINING SET METRICS ===\n")
print(train_results$metrics)

# Print results for test set
cat("\n=== TEST SET RESULTS ===\n")
print(test_results$confusion_matrix)

cat("\n=== TEST SET METRICS ===\n")
print(test_results$metrics)

# Create colored confusion matrix plots
train_plot <- plot_colored_confusion_matrix(train_pred, trainData$Subscription_Type, 
                                            "Training Set Confusion Matrix")
test_plot <- plot_colored_confusion_matrix(test_pred, testData$Subscription_Type, 
                                           "Decision Tree (J48) - Confusion Matrix")

# Display plots side by side
grid.arrange(train_plot, test_plot, ncol = 2)

# For training set
train_truth <- factor(trainData$Subscription_Type)
train_estimate <- factor(train_pred, levels = levels(train_truth))

train_metrics <- data.frame(
  Set = "Training",
  Accuracy = accuracy_vec(train_truth, train_estimate),
  Precision = precision_vec(train_truth, train_estimate),
  Recall = recall_vec(train_truth, train_estimate),
  F1_Score = f_meas_vec(train_truth, train_estimate)
)

# For test set
test_truth <- factor(testData$Subscription_Type)
test_estimate <- factor(test_pred, levels = levels(test_truth))

test_metrics <- data.frame(
  Set = "Test",
  Accuracy = accuracy_vec(test_truth, test_estimate),
  Precision = precision_vec(test_truth, test_estimate),
  Recall = recall_vec(test_truth, test_estimate),
  F1_Score = f_meas_vec(test_truth, test_estimate)
)

# Combine and print
all_metrics <- rbind(train_metrics, test_metrics)
print(all_metrics)


