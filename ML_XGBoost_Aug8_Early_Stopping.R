# Load libraries
library(xgboost)
library(caret)
library(dplyr)
library(Matrix)
library(pROC)

# Load dataset
df <- read.csv("../Dataset/streaming/train_streaming_subscription_churn.csv")

# Prepare target variable
df$subscription_type <- as.factor(df$subscription_type)
label_levels <- levels(df$subscription_type)
df$label_num <- as.numeric(df$subscription_type) - 1  # 0-based indexing for XGBoost

# Convert categorical variables to dummy variables
dummies <- dummyVars(~ location + payment_plan + payment_method + customer_service_inquiries, data = df)
dummy_data <- predict(dummies, newdata = df)

# Combine all features
features <- data.frame(
  Age = df$age,
  Days_Since_Joined_Date = df$Days_Since_Joined_Date,
  num_subscription_pauses = df$num_subscription_pauses,
  weekly_hours = df$weekly_hours,
  average_session_length = df$average_session_length,
  song_skip_rate = df$song_skip_rate,
  weekly_songs_played = df$weekly_songs_played,
  weekly_unique_songs = df$weekly_unique_songs,
  num_favorite_artists = df$num_favorite_artists,
  num_platform_friends = df$num_platform_friends,
  num_playlists_created = df$num_playlists_created,
  num_shared_playlists = df$num_shared_playlists,
  notifications_clicked = df$notifications_clicked,
  dummy_data
)

# Scale numeric features
preProc <- preProcess(features, method = c("center", "scale"))
features_scaled <- predict(preProc, features)

# Caret training setup
set.seed(123)
train_control <- trainControl(
  method = "cv",             # cross-validation
  number = 5,                # 5 folds
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Grid for hyperparameter tuning
grid <- expand.grid(
  nrounds = c(200, 400, 800),
  max_depth = c(3, 5, 7),
  eta = c(0.05, 0.1, 0.2),
  gamma = c(0, 1, 5),
  colsample_bytree = c(0.7, 0.9),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.7, 0.9)
)

# Train the model with caret
xgb_tuned <- train(
  x = features_scaled,
  y = df$subscription_type,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = grid,
  metric = "Accuracy"
)

# Best parameters
cat("\nBest Hyperparameters:\n")
print(xgb_tuned$bestTune)

# Accuracy from cross-validation
cat("\nBest CV Accuracy:", max(xgb_tuned$results$Accuracy), "\n")

# Feature importance
cat("\nTop Features:\n")
importance <- varImp(xgb_tuned)
print(importance)
plot(importance, top = 20)
