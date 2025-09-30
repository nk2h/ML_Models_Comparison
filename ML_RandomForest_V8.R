# Random Forest
# Load required libraries
library(caret)
library(randomForest)
library(dplyr)
library(yardstick)
library(ggplot2)
library(gridExtra)

# Load dataset
df <- read.csv("../Dataset/Test/Netflix Userbase.csv", header = TRUE)


df$Country <- as.factor(df$Country)
df$Gender <- as.factor(df$Gender)
df$Device <- as.factor(df$Device)
df$Start_Month <- as.factor(df$Start_Month)
df$Subscription_Type <- as.factor(df$Subscription_Type)


# Train-test split 
set.seed(42)
train_index <- createDataPartition(df$Subscription_Type, p = 0.8, list = FALSE)
train <- df[train_index, ]
test <- df[-train_index, ]



set.seed(42)

# Define tuning parameters
tuneGrid <- expand.grid(.mtry = c(2, 3, 4, 5)) 

# Train with cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

# Train Random Forest model
rf_model <- train(
  Subscription_Type ~ .,
  data = train,
  method = "rf",
  trControl = ctrl,
  tuneGrid = tuneGrid,
  ntree = 500,  # Number of trees
  importance = TRUE,  # Calculate variable importance
  metric = "Accuracy"
)

# Best tuning parameters
cat("\n=== BEST TUNING PARAMETERS ===\n")
print(rf_model$bestTune)

# Model summary
cat("\n=== RANDOM FOREST MODEL SUMMARY ===\n")
print(rf_model)

# Variable importance
cat("\n=== VARIABLE IMPORTANCE ===\n")
var_imp <- varImp(rf_model)
print(var_imp)

# Plot variable importance
plot(var_imp, main = "Random Forest - Variable Importance")

# Make predictions
train_pred <- predict(rf_model, newdata = train)
test_pred <- predict(rf_model, newdata = test)

# Function to calculate metrics (same format as previous models)
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

# Function to plot colored confusion matrix (same format as previous)
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
train_results <- calculate_metrics(train_pred, train$Subscription_Type)

# Calculate metrics for test set
test_results <- calculate_metrics(test_pred, test$Subscription_Type)

# Print results for training set
cat("\n=== TRAINING SET RESULTS ===\n")
print(train_results$confusion_matrix)

cat("\n=== TRAINING SET METRICS ===\n")
print(train_results$metrics)

# Print results for test set
cat("\n=== TEST SET RESULTS ===\n")
print(test_results$confusion_matrix)

cat("\n=== TEST SET METRICS ===\n")
print(test_results$metrics)

# Create colored confusion matrix plots
train_plot <- plot_colored_confusion_matrix(train_pred, train$Subscription_Type, 
                                            "Training Set Confusion Matrix")
test_plot <- plot_colored_confusion_matrix(test_pred, test$Subscription_Type, 
                                           "Random Forest - Confusion Matrix")

# Display plots side by side
grid.arrange(train_plot, test_plot, ncol = 2)


# For training set
train_truth <- factor(train$Subscription_Type)
train_estimate <- factor(train_pred, levels = levels(train_truth))

train_metrics <- data.frame(
  Set = "Training",
  Accuracy = accuracy_vec(train_truth, train_estimate),
  Precision = precision_vec(train_truth, train_estimate),
  Recall = recall_vec(train_truth, train_estimate),
  F1_Score = f_meas_vec(train_truth, train_estimate)
)

# For test set
test_truth <- factor(test$Subscription_Type)
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
