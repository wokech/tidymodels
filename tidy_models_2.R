# An Introduction to tidymodels
# Max Kuhn

## (1) Load the packages
library(tidymodels)

## (2) Load the data
data(Chicago, package = "modeldata")
head(Chicago)
dim(Chicago)
str(Chicago)

## (3) Select the appropriate features

chicago_rec <- recipe(ridership ~ ., data = Chicago) %>%
  step_date(date, features = c("dow", "month", "year")) %>%
  step_holiday(date) %>%
  step_rm(date) %>%
  step_dummy(all_nominal()) %>%
  step_normalize(all_predictors())

## (4) Fit the model with lin reg
linear_mod <- linear_reg(penalty = 0.1, mixture = 0.5) %>%
  set_engine("glmnet")

## (5) Create a modeling workflow

glmnet_wflow <-
  workflow() %>%
  add_model(linear_mod) %>%
  add_recipe(chicago_rec) # or add_formula() or add_variables()

## (6) Fit and predict

glmnet_fit <- fit(glmnet_wflow, data = Chicago)
predict(glmnet_fit, Chicago %>% slice(1:7))

## (7) Model tuning

linear_mod <-
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

glmnet_wflow <-
  glmnet_wflow %>%
  update_model(linear_mod)

## (8) Resampling and Grid Search

chicago_rs <-
  sliding_period(
    Chicago,
    date,
    period = "month",
    lookback = 14 * 12,
    assess_stop = 1
  )
chicago_rs

install.packages("doMC", repos="http://R-Forge.R-project.org")

library(doMC)
registerDoMC(cores = parallel::detectCores())
set.seed(29)
glmnet_tune <-
  glmnet_wflow %>%
  tune_grid(chicago_rs, grid = 10)

show_best(glmnet_tune, metric = "rmse")

collect_metrics(glmnet_tune) %>% slice(1:10)

Next steps

There are functions to plot the results, substitute the best parameters for the tune() placeholders, fit the final model,
measure the test set performance