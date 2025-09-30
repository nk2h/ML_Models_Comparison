# XGBoost
# Load required libraries
library(caret)
library(xgboost)
library(dplyr)
library(yardstick)
library(ggplot2)
library(gridExtra)
library(Matrix) 

# Load dataset
df <- read.csv("../Dataset/Netflix Userbase_bk2.csv", header = TRUE)

# Data preprocessing
df$Country <- as.factor(df$Country)
df$Gender <- as.factor(df$Gender)
df$Device <- as.factor(df$Device)
df$Start_Month <- as.factor(df$Start_Month)
df$Subscription_Type <- as.factor(df$Subscription_Type)


# Train-test split (70-30 split as in previous models)
set.seed(42)
train_index <- createDataPartition(df$Subscription_Type, p = 0.8, list = FALSE)
train <- df[train_index, ]
test <- df[-train_index, ]

# Convert factors to numeric for XGBoost
prepare_xgb_data <- function(data) {
  # Create model matrix (one-hot encoding for categorical variables)
  features <- model.matrix(~ . -1, data = data %>% select(-Subscription_Type))
  
  # Convert to matrix
  features_matrix <- as.matrix(features)
  
  # Convert target variable to numeric (0-indexed)
  target <- as.numeric(data$Subscription_Type) - 1
  
  return(list(features = features_matrix, target = target))
}

# Prepare training and test data
train_prepared <- prepare_xgb_data(train)
test_prepared <- prepare_xgb_data(test)

# Set parameters for multi-class classification
num_class <- length(unique(df$Subscription_Type))
cat("Number of classes:", num_class, "\n")

# Define parameter grid 
param_grid <- expand.grid(
  nrounds = c(100, 200),           
  max_depth = c(3, 6, 9),          
  eta = c(0.01, 0.1, 0.3),        
  gamma = c(0, 1),                 
  colsample_bytree = c(0.8, 1.0),  
  min_child_weight = c(1),         
  subsample = c(0.8, 1.0)          
)

cat("\n=== TRAINING XGBOOST MODEL ===\n")
set.seed(42)

# Train XGBoost model with cross-validation
xgb_model <- train(
  x = train_prepared$features,
  y = factor(train$Subscription_Type),
  method = "xgbTree",
  trControl = trainControl(
    method = "cv",
    number = 10,
    classProbs = TRUE,
    summaryFunction = multiClassSummary,
    verboseIter = TRUE
  ),
  tuneGrid = param_grid,
  metric = "Accuracy",
  nthread = 4,  # Use 4 cores for parallel processing
  verbosity = 0
)

# Best tuning parameters
cat("\n=== BEST TUNING PARAMETERS ===\n")
print(xgb_model$bestTune)

# Model summary
cat("\n=== XGBOOST MODEL SUMMARY ===\n")
print(xgb_model)

# Variable importance
cat("\n=== VARIABLE IMPORTANCE ===\n")
var_imp <- varImp(xgb_model)
print(var_imp)

# Plot variable importance
plot(var_imp, main = "XGBoost - Variable Importance")

# Make predictions
train_pred <- predict(xgb_model, newdata = train_prepared$features)
test_pred <- predict(xgb_model, newdata = test_prepared$features)

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
                                           "XGBoost - Confusion Matrix")

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


