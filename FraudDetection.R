############################################################
# CREDIT CARD FRAUD DETECTION PROJECT IN R
# Author: Triston Aloyssius Marta
############################################################

# ======================
# 1. Load Libraries
# ======================
options(repos = c(CRAN = "https://cloud.r-project.org"))

required <- c("data.table", "dplyr", "ggplot2", "caret", "smotefamily",
              "randomForest", "xgboost", "pROC", "rpart", "rpart.plot",
              "nnet", "lightgbm")

install_if_missing <- function(pkgs){
  to_install <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
  if(length(to_install)) install.packages(to_install, dependencies = TRUE)
}
install_if_missing(required)
lapply(required, library, character.only = TRUE)

# ======================
# 2. Load Dataset
# ======================
data <- fread("creditcard.csv")
cat("Dataset loaded with", nrow(data), "rows and", ncol(data), "columns\n")

# Explore
str(data)
summary(data)
table(data$Class)

# ======================
# 3. Preprocessing
# ======================
set.seed(123)
data$Amount <- scale(data$Amount)
data$Time <- scale(data$Time)

# Split Train/Test
trainIndex <- createDataPartition(data$Class, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)

# ======================
# 4. Handle Imbalance (SMOTE)
# ======================
set.seed(42)
X <- as.data.frame(train[, -which(names(train) == "Class"), with = FALSE])
y <- as.factor(train$Class)
smote_out <- SMOTE(X, y, K = 5, dup_size = 0)

smote_train <- smote_out$data
colnames(smote_train)[ncol(smote_train)] <- "Class"
smote_train$Class <- as.factor(smote_train$Class)
cat("After SMOTE:", table(smote_train$Class), "\n")

# ======================
# 5. Model Training
# ======================

## Logistic Regression
log_model <- glm(Class ~ ., data = smote_train, family = binomial)
log_pred <- predict(log_model, newdata = test, type = "response")
log_class <- ifelse(log_pred > 0.5, 1, 0)

## Decision Tree
tree_model <- rpart(Class ~ ., data = smote_train, method = "class")
tree_pred <- predict(tree_model, test, type = "class")

## Random Forest
rf_model <- randomForest(Class ~ ., data = smote_train, ntree = 100, mtry = 6)
rf_pred <- predict(rf_model, test)

## XGBoost
smote_train <- as.data.table(smote_train)
test <- as.data.table(test)
train_y <- as.numeric(as.character(smote_train$Class))
test_y <- as.numeric(as.character(test$Class))
train_y[is.na(train_y)] <- 0
test_y[is.na(test_y)] <- 0

common_cols <- intersect(names(smote_train)[!names(smote_train) %in% "Class"],
                         names(test)[!names(test) %in% "Class"])

train_x <- as.matrix(smote_train[, ..common_cols])
test_x  <- as.matrix(test[, ..common_cols])

train_matrix <- xgb.DMatrix(data = train_x, label = train_y)
test_matrix  <- xgb.DMatrix(data = test_x,  label = test_y)

params <- list(objective = "binary:logistic", eval_metric = "auc",
               max_depth = 6, eta = 0.1, subsample = 0.8, colsample_bytree = 0.8)

set.seed(42)
xgb_model <- xgb.train(params = params, data = train_matrix, nrounds = 100, verbose = 0)
xgb_pred <- predict(xgb_model, test_matrix)
xgb_class <- ifelse(xgb_pred > 0.5, 1, 0)

## Neural Network
nn_model <- nnet(Class ~ ., data = smote_train, size = 5, maxit = 500, decay = 0.01)
nn_pred <- predict(nn_model, test, type = "raw")
nn_class <- ifelse(nn_pred > 0.5, 1, 0)

## LightGBM
train_y <- as.numeric(as.character(smote_train$Class))
test_y <- as.numeric(as.character(test$Class))
train_y[is.na(train_y)] <- 0
test_y[is.na(test_y)] <- 0

common_cols <- intersect(names(smote_train)[!names(smote_train) %in% "Class"],
                         names(test)[!names(test) %in% "Class"])
train_x <- as.matrix(smote_train[, ..common_cols])
test_x  <- as.matrix(test[, ..common_cols])

stopifnot(nrow(train_x) == length(train_y))
stopifnot(nrow(test_x) == length(test_y))

train_lgb <- lgb.Dataset(data = train_x, label = train_y)
params_lgb <- list(objective = "binary", metric = "auc", learning_rate = 0.05,
                   num_leaves = 31, feature_fraction = 0.8, bagging_fraction = 0.8,
                   bagging_freq = 5)

set.seed(42)
lgb_model <- lgb.train(params = params_lgb, data = train_lgb, nrounds = 100, verbose = -1)
lgb_pred <- predict(lgb_model, test_x)
lgb_class <- ifelse(lgb_pred > 0.5, 1, 0)

# ======================
# 6. Evaluation
# ======================
get_metrics <- function(true, pred_prob, pred_class) {
  auc <- roc(true, pred_prob)$auc
  cm <- confusionMatrix(as.factor(pred_class), as.factor(true), positive = "1")
  acc <- cm$overall["Accuracy"]
  recall <- cm$byClass["Recall"]
  precision <- cm$byClass["Precision"]
  f1 <- cm$byClass["F1"]
  data.frame(AUC = auc, Accuracy = acc, Recall = recall, Precision = precision, F1 = f1)
}

results <- rbind(
  cbind(Model = "Logistic Regression", get_metrics(test$Class, log_pred, log_class)),
  cbind(Model = "Decision Tree", get_metrics(test$Class, as.numeric(tree_pred), tree_pred)),
  cbind(Model = "Random Forest", get_metrics(test$Class, as.numeric(rf_pred), rf_pred)),
  cbind(Model = "XGBoost", get_metrics(test$Class, xgb_pred, xgb_class)),
  cbind(Model = "Neural Network", get_metrics(test$Class, nn_pred, nn_class)),
  cbind(Model = "LightGBM", get_metrics(test$Class, lgb_pred, lgb_class))
)

print(results)

# Visual Comparison
ggplot(results, aes(x = Model, y = AUC, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison - AUC Scores") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ======================
# 7. Save Best Model
# ======================
best_model <- rf_model
saveRDS(best_model, "best_model_rf.rds")


