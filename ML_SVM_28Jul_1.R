# Load required libraries
library(e1071)
library(caret)
library(tidyverse)
library(pROC)        # For multiclass ROC-AUC
library(MLmetrics)   # For F1-score, Precision, Recall
library(ggplot2)
library(reshape2)

# Load dataset
df <- read.csv("../Dataset/Netflix Userbase.csv")

# Convert categorical variables to factors
df$Country <- as.factor(df$Country)
df$Gender <- as.factor(df$Gender)
df$Device <- as.factor(df$Device)
df$Subscription_Type <- as.factor(df$Subscription_Type)

View(df)

# Identify numeric columns for scaling
numeric_cols <- c("Days_Since_Joined_Date", "Age")

# Scale numeric features
preProc <- preProcess(df[, numeric_cols], method = c("center", "scale"))
df[, numeric_cols] <- predict(preProc, df[, numeric_cols])

# Train-test split (stratified)
set.seed(123)
trainIndex <- createDataPartition(df$Subscription_Type, p = 0.7, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

#-------------------------
# Tune SVM with radial kernel
#-------------------------
set.seed(123)
tuned_model <- tune(svm, Subscription_Type ~ ., data = trainData,
                    kernel = "radial", probability = TRUE, 
                    ranges = list(cost = 2^(0:5), gamma = 2^(-5:-1)))

# Best model
best_model <- tuned_model$best.model

# Predict on test data
pred <- predict(best_model, testData)

# Make sure prediction and true labels have same factor levels
pred <- factor(pred, levels = levels(df$Subscription_Type))
true_labels <- factor(testData$Subscription_Type, levels = levels(df$Subscription_Type))

#-------------------------
# Confusion Matrix and Accuracy
#-------------------------
cm <- confusionMatrix(pred, true_labels)
print(cm)

# Metrics: Precision, Recall, F1
precision <- diag(cm$table) / colSums(cm$table)
recall <- diag(cm$table) / rowSums(cm$table)
f1 <- 2 * ((precision * recall) / (precision + recall))
metrics_df <- data.frame(
  Class = levels(true_labels),
  Precision = round(precision, 3),
  Recall = round(recall, 3),
  F1_Score = round(f1, 3)
)
print(metrics_df)

# Confusion Matrix Heatmap
cm_df <- as.data.frame(as.table(cm$table))
colnames(cm_df) <- c("Actual", "Predicted", "Freq")
ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix Heatmap")

# Barplot of Precision, Recall, F1
metrics_long <- melt(metrics_df, id.vars = "Class")
ggplot(metrics_long, aes(x = Class, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Precision, Recall, F1-Score per Class",
       y = "Score", x = "Class") +
  scale_fill_manual(values = c("skyblue", "orange", "darkgreen")) +
  theme_minimal()

# ROC-AUC (one-vs-all)
# Refit model with probability = TRUE
svm_prob_model <- svm(Subscription_Type ~ ., data = trainData, kernel = "radial",
                      cost = best_model$cost, gamma = best_model$gamma,
                      probability = TRUE)

# Predict probabilities
pred_prob <- attr(predict(svm_prob_model, testData, probability = TRUE), "probabilities")

# Plot ROC Curves
plot(roc(true_labels == levels(true_labels)[1], pred_prob[,1]), col = 1,
     main = "Multiclass ROC Curves")
for (i in 2:length(levels(true_labels))) {
  lines(roc(true_labels == levels(true_labels)[i], pred_prob[,i]), col = i)
}
legend("bottomright", legend = levels(true_labels), col = 1:length(levels(true_labels)), lwd = 2)

# AUC values
auc_values <- sapply(1:ncol(pred_prob), function(i) {
  roc_obj <- roc(true_labels == levels(true_labels)[i], pred_prob[, i])
  auc(roc_obj)
})
names(auc_values) <- levels(true_labels)
print(round(auc_values, 3))



