
#### Libraries ####
library(tidymodels)
library(tidyverse)
tidymodels_prefer()


#### Feature Engineering ####

# Lood the data 
bank_full <- read_delim("Data/data_raw/bank-full.csv",delim = ";", escape_double = FALSE, trim_ws = TRUE)

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

# Make a basic recipe
recipe_basic <- recipe(y ~ .,data = train_data)%>%
  
  # Remove near zero variance features
  step_nzv(all_nominal_predictors()) %>%
  
  # Transform all numerical featues to reduce numerical skewness
  step_YeoJohnson(all_numeric_predictors()) %>%
  
  # Center and Scale all numerical features
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  
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

## SVM Radial 
svm_spec <- svm_rbf(cost = tune(),rbf_sigma = tune())%>%
  set_engine("kernlab") %>%
  set_mode("classification")

## Random Forest 
rf_spec <- rand_forest(mtry = tune(),trees = 200,min_n = tune())%>%
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

# SVM Radial
svm_workflow <- 
  workflow() %>%
  add_recipe(recipe = recipe_basic)%>%
  add_model(spec = svm_spec)

# Random Forest 
rf_workflow <- 
  workflow() %>%
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
    tree_depth = tree_depth(c(5,15)),
    min_n = min_n(c(20,100))
  )

# SVM Radial 
svm_params <- 
  svm_workflow %>%
  extract_parameter_set_dials()%>%
  update(
    cost = cost(c(-10,5)),
    rbf_sigma = rbf_sigma(c(-10,0))
  )

# RF
rf_params <- 
  rf_workflow %>%
  extract_parameter_set_dials()%>%
  finalize(x = train_data)%>%
  update(
    mtry = mtry(c(1,16)),
    min_n = min_n(c(20,100))
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

# SVM Radianl LHC Grid
svm_grid <- svm_params %>%
  grid_space_filling()

# RF LHC Grid
rf_grid <- rf_params %>%
  grid_space_filling()

# XGB LHC Grid
xgb_grid <- xgb_params %>%
  grid_space_filling()

#### Rasample and Evaluationg Metric ####

## Create a Cross Validation Resampling Method
resample_cv <- vfold_cv(data = train_data,v = 10)

## Metric 
metric <- metric_set(roc_auc)

#### Initial Search ####

#<> Fit the log regression
log_regression_fit <- 
  log_regression_workflow %>%
  fit_resamples(
    resamples = resample_cv,
    metrics = metric)

# Decision trees
d_tree_initial <-
  d_tree_workflow %>%
  tune_grid(
    resamples = resample_cv,
    metrics = metric,
    grid = d_tree_grid)

d_t








