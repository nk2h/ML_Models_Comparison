# Load libraries
library(xgboost)
library(caret)
library(dplyr)
library(Matrix)
library(pROC)

# Load dataset
df <- read.csv("../Dataset/Netflix Userbase_bk.csv")

# Prepare target variable
df$Subscription_Type <- as.factor(df$Subscription_Type)
label_levels <- levels(df$Subscription_Type)
df$label_num <- as.numeric(df$Subscription_Type) - 1  # 0-based indexing for XGBoost

# Convert categorical variables to dummy variables
dummies <- dummyVars(~ Country + Gender + Device, data = df)
dummy_data <- predict(dummies, newdata = df)

# Combine all features
features <- data.frame(df$Days_Since_Joined_Date, df$Age, dummy_data)
colnames(features)[1:2] <- c("Days_Since_Joined_Date", "Age")

# Scale numeric features
preProc <- preProcess(features, method = c("center", "scale"))
features_scaled <- predict(preProc, features)

# Final dataset
final_data <- data.frame(features_scaled, label = df$label_num)

# Split data into training and validation sets
set.seed(123)
train_index <- createDataPartition(final_data$label, p = 0.75, list = FALSE)
train_df <- final_data[train_index, ]
valid_df <- final_data[-train_index, ]

# Create DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_df[, -ncol(train_df)]), label = train_df$label)
dvalid <- xgb.DMatrix(data = as.matrix(valid_df[, -ncol(valid_df)]), label = valid_df$label)

# Set parameter list for multiclass classification
num_class <- length(unique(final_data$label))
params <- list(
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = num_class,
  eta = 0.1,
  max_depth = 2,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train with early stopping
watchlist <- list(train = dtrain, eval = dvalid)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 800,
  watchlist = watchlist,
  early_stopping_rounds = 20,
  print_every_n = 10,
  maximize = FALSE
)

# Predict on validation set
pred_prob <- predict(xgb_model, as.matrix(valid_df[, -ncol(valid_df)]))
pred_matrix <- matrix(pred_prob, ncol = num_class, byrow = TRUE)
pred_label <- max.col(pred_matrix) - 1  # Convert to 0-based label

# Confusion Matrix and Accuracy
true_label <- valid_df$label
conf_mat <- confusionMatrix(factor(pred_label, levels = 0:(num_class-1), labels = label_levels),
                            factor(true_label, levels = 0:(num_class-1), labels = label_levels))

print(conf_mat)

# Precision, Recall, F1
cm_table <- conf_mat$table
precision <- diag(cm_table) / colSums(cm_table)
recall <- diag(cm_table) / rowSums(cm_table)
f1 <- 2 * ((precision * recall) / (precision + recall))

metrics_df <- data.frame(
  Class = label_levels,
  Precision = round(precision, 3),
  Recall = round(recall, 3),
  F1_Score = round(f1, 3)
)
print(metrics_df)

