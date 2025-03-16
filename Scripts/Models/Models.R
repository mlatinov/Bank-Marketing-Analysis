
#### Libraries ####
library(tidymodels)
library(tidyverse)
library(themis)
library(doParallel)
library(finetune)
library(future)
library(pROC)
library(PRROC)

tidymodels_prefer()

num_cores <- parallel::detectCores()  # Detect the number of cores on your machine

# Set up parallel processing using `future`
plan(multisession, workers = min(num_cores, 4))

#### Feature Engineering ####

# Lood the data 
bank_full <- read_delim("bank-full.csv",delim = ";", escape_double = FALSE, trim_ws = TRUE)

# Remove duration 
bank_full <- bank_full %>% select(-duration)

# Manually encode it as 0 = no 1 = yes
bank_full$y <- ifelse(test = bank_full$y == "no" ,yes = 0,no = 1 )

# Change the outcome into factor
bank_full$y <- as.factor(bank_full$y)


## Split the data 
bank_split <- initial_split(data = bank_full,prop = 0.8,strata = y)

# Training data 
train_data <- training(bank_split)

# Testing data 
testing_data <- testing(bank_split)

## Recipe ##

## Add Case Weights with frequency_weights
train_data <- train_data %>%
  mutate(weights = hardhat::frequency_weights(as.numeric(y)))

# Make a basic recipe
recipe_basic <- recipe(y ~ .,data = train_data)%>%
  
  # Remove near zero variance features
  step_nzv(all_nominal_predictors()) %>%
  
  # Transform all numerical features to reduce numerical skewness
  step_YeoJohnson(all_numeric_predictors(),-weights) %>%
  
  # Center and Scale all numerical features
  step_center(all_numeric_predictors(),-weights) %>%
  step_scale(all_numeric_predictors(),-weights) %>%
  
  # Dummy encode all nominal predictors
  step_dummy(all_nominal_predictors()) 
  
#### Define models ####

## Log Regression Baseline
log_regression <- logistic_reg()%>%
  set_engine("glm")%>%
  set_mode("classification")

## Decision trees
d_tree <- decision_tree(tree_depth = tune(),min_n = tune())%>%
  set_engine("rpart")%>%
  set_mode("classification")

## SVM Poly
svm_spec <- svm_poly(cost = tune(),degree = tune())%>%
  set_engine("kernlab")%>%
  set_mode("classification")

## Random Forest 
rf_spec <- rand_forest(mtry = tune(),trees = 300,min_n = tune())%>%
  set_engine("ranger")%>%
  set_mode("classification")

## XGB
xgb_spec <- boost_tree(
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  learn_rate = tune(),
  sample_size = tune(),
  tree_depth = tune()
  ) %>%
  set_engine("xgboost")%>%
  set_mode("classification")

#### Create Workflows ####

# Log Regression Workflow 
log_regression_workflow <-
  workflow() %>%
  add_recipe(recipe = recipe_basic)%>%
  add_model(spec = log_regression)

# Decision trees
d_tree_workflow <- 
  workflow() %>%
  add_recipe(recipe = recipe_basic)%>%
  add_model(spec = d_tree)

# SVM Poly
svm_workflow <- 
  workflow() %>%
  add_recipe(recipe = recipe_basic)%>%
  add_model(spec = svm_spec)

# Random Forest 
rf_workflow <- 
  workflow() %>%
  add_case_weights(weights) %>%
  add_recipe(recipe = recipe_basic) %>%
  add_model(spec = rf_spec)

# XGB 
xgb_worklow <- 
  workflow() %>%
  add_recipe(recipe = recipe_basic)%>%
  add_model(spec = xgb_spec)

#### Set parameter ####

# Decision trees
d_tree_params <- 
  d_tree_workflow %>%
  extract_parameter_set_dials()%>%
  update(
    tree_depth = tree_depth(c(5,20)),
    min_n = min_n(c(10,150))
  )

# SVM Poly
svm_params <- 
  svm_workflow %>%
  extract_parameter_set_dials()%>%
  update(
    cost = cost(c(-5,5)),
    degree = degree(c(1,3))
  )

# RF
rf_params <- 
  rf_workflow %>%
  extract_parameter_set_dials()%>%
  finalize(x = train_data)%>%
  update(
    mtry = mtry(c(1,16)),
    min_n = min_n(c(5,100))
  )

# XGBoost
xgb_params <- 
  xgb_worklow %>%
  extract_parameter_set_dials()%>%
  finalize(x = train_data)%>%
  update(
    mtry = mtry(c(1,16)),
    trees = trees(c(1000,2000)),
    min_n = min_n(c(20,100)),
    tree_depth = tree_depth(c(5,15)),
    learn_rate = learn_rate(c(-10,-1)),
    sample_size = sample_size(c(0,1))
  )
  
#### Start Grid  ####

# Decision trees LHC Grid
d_tree_grid <- d_tree_params %>%
  grid_space_filling()

# SVM Poly LHC Grid
svm_grid <- svm_params %>%
  grid_space_filling()

# RF LHC Grid
rf_grid <- rf_params %>%
  grid_space_filling()

# XGB LHC Grid
xgb_grid <- xgb_params %>%
  grid_space_filling()

#### Rasample and Evaluation Metric ####

## Create a Cross-Validation with stratification 
resample_cv <- vfold_cv(data = train_data,v = 10,strata = y)

## Create a Monte Carlo cross-validation with stratification
resample_mccv <- mc_cv(data = train_data,prop = 0.8,times = 10,strata = y)

## Metric 
metric <- metric_set(pr_auc,roc_auc,accuracy)

# Control resample 
control_resample <- control_resamples(verbose = TRUE,parallel_over = "everything")

#### Initial Search ####

#<> Fit the log regression
log_regression_fit <- 
  log_regression_workflow %>%
  fit_resamples(
    resamples = resample_cv,
    metrics = metric
    )
log_regression_fit$.metrics
  
# Decision trees
d_tree_initial <-
  d_tree_workflow %>%
  tune_grid(
    resamples = resample_cv,
    metrics = metric,
    grid = d_tree_grid
    )

# SVM Poly 
svm_initial <- 
  svm_workflow %>%
  tune_grid(
    resamples = resample_mccv,
    grid = svm_grid,
    metrics = metric)

# RF
rf_initial <- 
  rf_workflow %>%
  tune_grid(
    resamples = resample_mccv,
    grid = rf_grid,
    metrics = metric,
    control = control_resample)
rf_initial$.metrics

# XGB 
xbg_initial <- 
  xgb_worklow %>%
  tune_grid(
    resamples = resample_mccv,
    grid = xgb_grid,
    metrics = metric,
    control = control_resample
  )
xbg_initial$.metrics

#### Bayesian optimization ####

# Define a BO control
control_bo <- control_bayes(verbose = TRUE,parallel_over = "everything")

# Metric to optimize
metric_bo <- metric_set(pr_auc,roc_auc)

# Execute BO on RF
rf_bo <- 
  rf_workflow %>%
  tune_bayes(
    resamples = resample_mccv,
    initial = rf_initial,
    param_info = rf_params,
    iter = 10,
    metrics = metric_bo,
    control = control_bo
    )

# Best result 
best_rf_bo <- select_best(rf_bo,metric = "roc_auc")

#### Simulated Annealing ####

# Define SA control 
control_sa <- control_sim_anneal(
  parallel_over = "everything",
  verbose = TRUE,
  no_improve = 10
  )

# Metric to optimize
metric_sa <- metric_set(pr_auc)

# Execute SA on RF
rf_sa <- 
  rf_workflow %>%
  tune_sim_anneal(
    resamples = resample_mccv,
    param_info = rf_params,
    iter = 10,
    metrics = metric_sa,
    initial = rf_initial,
    control = control_sa
    )

# Best results
best_sa_rf <- select_best(rf_sa,metric = "pr_auc")

# Function to compute metrics and plot AUC curve and Confusion Matrix
md_fit_eval <- function(workflow, best_params, train_data, testing_data) {
  
  # Convert weights to numeric
  train_data <- train_data %>%
    mutate(weights = as.numeric(weights))
  
  # Finalize the workflow with the best params
  final_md <- finalize_workflow(workflow, best_params)
  
  # Fit the model with weights
  final_model <- workflows::fit(object = final_md, data = train_data)
  
  # Predictions on the test data
  predictions <- predict(final_model, testing_data, type = "prob") %>%
    bind_cols(predict(final_model, testing_data, type = "class")) %>%
    bind_cols(testing_data) 
  
  # Extract positive class probabilities and true labels
  prediction_pos <- predictions$.pred_1
  true_labels <- predictions$y
  
  # Compute AUC
  roc_result <- roc(true_labels, prediction_pos)
  auc_value <- auc(roc_result)
  
  # Compute PR AUC
  pr_auc_value <- pr_auc(predictions, truth = y, .pred_1)$.estimate
  
  # Convert probabilities to class predictions (threshold = 0.5)
  predictions <- predictions %>%
    mutate(predicted_class = factor(ifelse(.pred_1 >= 0.5, "1", "0"), levels = levels(true_labels)))
  
  # Accuracy
  accuracy_value <- accuracy(predictions, truth = y, estimate = predicted_class)$.estimate
  
  # Confusion Matrix
  conf_matrix <- conf_mat(predictions, truth = y, estimate = predicted_class)
  
  # Plot AUC curve
  auc_plot <- ggroc(roc_result) + ggtitle("ROC Curve")
  
  # Plot confusion matrix
  conf_matrix_plot <- conf_matrix %>%
    autoplot(type = "heatmap") + ggtitle("Confusion Matrix")
  
  # Return results
  return(list(
    auc = auc_value,
    pr_auc = pr_auc_value,
    accuracy = accuracy_value,
    auc_plot = auc_plot,
    conf_matrix_plot = conf_matrix_plot
  ))
}

# Evaluate RF Model 
md_fit_eval(
  workflow = rf_workflow,
  best_params = best_rf_bo,
  train_data = train_data,
  testing_data = testing_data
  )


stopCluster(cl)
registerDoSEQ()  # Switch back to sequential mode















