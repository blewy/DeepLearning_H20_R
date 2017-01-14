# Example 1: Optimization
## Set Pred = 0, as placeholder
Test_Fun <- function(x) {
  list(Score = exp(-(x - 2)^2) + exp(-(x - 6)^2/10) + 1/ (x^2 + 1),
       Pred = 0)
}
## Set larger init_points and n_iter for better optimization result
OPT_Res <- BayesianOptimization(Test_Fun,
                                bounds = list(x = c(1, 3)),
                                init_points = 2, n_iter = 1,
                                acq = "ucb", kappa = 2.576, eps = 0.0,
                                verbose = TRUE)

OPT_Res


# Example 2: Parameter Tuning
library(xgboost)
data(agaricus.train, package = "xgboost")
dtrain <- xgb.DMatrix(agaricus.train$data,
                      label = agaricus.train$label)
cv_folds <- KFold(agaricus.train$label, nfolds = 5,
                  stratified = TRUE, seed = 0)
xgb_cv_bayes <- function(max.depth, min_child_weight, subsample) {
  cv <- xgb.cv(params = list(booster = "gbtree", eta = 0.01,
                             max_depth = max.depth,
                             min_child_weight = min_child_weight,
                             subsample = subsample, colsample_bytree = 0.3,
                             lambda = 1, alpha = 0,
                             objective = "binary:logistic",
                             eval_metric = "auc"),
               data = dtrain, nround = 100,
               folds = cv_folds, prediction = TRUE, showsd = TRUE,
               early_stop_round = 5, maximize = TRUE, verbose = 0)
  list(Score = max(cv$evaluation_log[,test_auc_mean]),
       Pred = cv$pred)
}
OPT_Res <- BayesianOptimization(xgb_cv_bayes,bounds = list(max.depth = c(2L, 6L),
                                                           min_child_weight = c(1L, 10L),
                                                           subsample = c(0.5, 0.8)),
                                init_grid_dt = NULL, init_points = 10, n_iter = 20,
                                acq = "ucb", kappa = 2.576, eps = 0.0,
                                verbose = TRUE)

                                
                                
