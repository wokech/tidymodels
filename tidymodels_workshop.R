## tidymodels Workshop

# Day One*******************************

# (1) Introduction ###############################################################

## Load the required packages
library(tidymodels)

## Important packages for ML modeling

install.packages(c("Cubist", "DALEXtra", "doParallel", "earth", "embed", 
                   "forcats", "lme4", "parallelly", "ranger", "remotes", "rpart", 
                   "rpart.plot", "rules", "stacks", "tidymodels",
                   "vetiver", "xgboost"))

remotes::install_github("topepo/ongoal@hockeyR")

# (2) Your data budget ###########################################################

library(tidymodels)

data("tree_frogs", package = "stacks")
tree_frogs <- tree_frogs %>%
  mutate(t_o_d = factor(t_o_d),
         age = age / 86400) %>%
  filter(!is.na(latency)) %>%
  select(-c(clutch, hatched))

# Summary: Data on tree frog hatching
# N = 572
# A numeric outcome, latency

# 4 other variables
# treatment, reflex, and t_o_d are nominal predictors
# age is a numeric predictor

# Data splitting and spending
# For machine learning, we typically split data into training and test sets:
  
## The training set is used to estimate model parameters.
## The test set is used to find an independent assessment of model performance.

## The more data we spend, the better estimates we will get. 

## Spending too much data in training prevents us from computing a good 
## assessment of predictive performance.

## Spending too much data in testing prevents us from computing a good 
## estimate of model parameters.

set.seed(123)
frog_split <- initial_split(tree_frogs)
frog_split

## Accessing the data

frog_train <- training(frog_split)
frog_test <- testing(frog_split)

set.seed(123)
frog_split <- initial_split(tree_frogs, prop = 0.8)
frog_train <- training(frog_split)
frog_test <- testing(frog_split)

nrow(frog_train)

nrow(frog_test)

frog_train %>%
  ggplot(aes(x=latency)) + 
  geom_histogram(bins = 20)

frog_train %>%
  ggplot(aes(latency, treatment, fill = treatment)) + 
  geom_boxplot(alpha = 0.6, show.legend = FALSE)

frog_train %>%
  ggplot(aes(latency, reflex, fill = reflex)) + 
  geom_boxplot(alpha = 0.3, show.legend = FALSE)

frog_train %>%
  ggplot(aes(age, latency, color = reflex)) + 
  geom_point(alpha = 0.8, size = 1)

## Split smarter

## Stratified sampling would split within each quartile
## Allows for better splits with skewed data

set.seed(123)
frog_split <- initial_split(tree_frogs, prop = 0.8, strata = latency)
frog_split

# (3) What makes a model? ####################################################

## Fitting a linear model in R

# lm for linear model
# glm for generalized linear model (e.g. logistic regression)
# glmnet for regularized regression
# keras for regression using TensorFlow
# stan for Bayesian regression
# spark for large data sets

## To specify a model

## Search parsnip models: https://www.tidymodels.org/find/parsnip/

# Choose a model
# Specify an engine
# Set the mode

linear_reg() %>%
  set_engine("glmnet")

decision_tree() %>% 
  set_mode("regression")

## Linear regression

# Outcome modeled as linear combination of predictors: latency=β0+β1⋅age+ϵ
# Find a line that minimizes the mean squared error (MSE)

## Decision trees

# Series of splits or if/then statements based on predictors
# First the tree grows until some condition is met (maximum depth, no more data)
# Then the tree is pruned to reduce its complexity

# All models are wrong, but some are useful!

## A model workflow

## Workflows bind preprocessors and models

## Data + Predictors --> Workflow (PCA + Least Sq Estimation) --> Fitted Model

## Why a workflow()?

# Workflows handle new data better than base R tools in terms of new factor levels.
# You can use other preprocessors besides formulas (more on feature engineering tomorrow!).
# They can help organize your work when working with multiple models.
# Most importantly, a workflow captures the entire modeling process: 
# fit() and predict() apply to the preprocessing steps in addition to the actual model fit.

## Example

install.packages("pec")
library(pec)

tree_spec <-
  decision_tree() %>% 
  set_mode("regression")

tree_spec %>% 
  fit(latency ~ ., data = frog_train)

tree_spec <-
  decision_tree() %>% 
  set_mode("regression")

tree_wflow <- workflow() %>%
  add_formula(latency ~ .) %>%
  add_model(tree_spec) %>%
  fit(data = frog_train) 


tree_spec <-
  decision_tree() %>% 
  set_mode("regression")

tree_wflow <- workflow(latency ~ ., tree_spec) %>% 
  fit(data = frog_train)

## Predict with your model

tree_spec <-
  decision_tree() %>% 
  set_mode("regression")

tree_fit <-
  workflow(latency ~ ., tree_spec) %>% 
  fit(data = frog_train) 

predict(tree_fit, new_data = frog_test)

augment(tree_fit, new_data = frog_test)

## The tidymodels prediction guarantee!

# The predictions will always be inside a tibble
# The column names and types are unsurprising and predictable
# The number of rows in new_data and the output are the same

library(rpart.plot)
tree_fit %>%
  extract_fit_engine() %>%
  rpart.plot(roundint = FALSE)

## Understand your model
# How do you understand your new tree_fit model?

# You can use your fitted workflow for model and/or prediction explanations:

# Overall variable importance, such as with the vip package
# Flexible model explainers, such as with the DALEXtra package

###### Never predict() with any extracted components! ############

## Deploy your model ( https://vetiver.rstudio.com)

## How do you use your new tree_fit model in production?

library(vetiver)
v <- vetiver_model(tree_fit, "frog_hatching")
v

library(plumber)
pr() %>%
  vetiver_api(v)


# (4) Evaluating models #########################################################

## Metrics for model performance

augment(tree_fit, new_data = frog_test) %>%
  metrics(latency, .pred)

# RMSE: difference between the predicted and observed values
# R2: squared correlation between the predicted and observed values
# MAE: similar to RMSE, but mean absolute error

augment(tree_fit, new_data = frog_test) %>%
  rmse(latency, .pred)

augment(tree_fit, new_data = frog_test) %>%
  group_by(reflex) %>%
  rmse(latency, .pred)

frog_metrics <- metric_set(rmse, msd)
augment(tree_fit, new_data = frog_test) %>%
  frog_metrics(latency, .pred)

####### OVERFITTING IS DANGEROUS #########

# We call this “resubstitution” or “repredicting the training set”

tree_fit %>%
  augment(frog_train)

# We call this a “resubstitution estimate”

tree_fit %>%
  augment(frog_train) %>%
  rmse(latency, .pred)

# With the training set

tree_fit %>%
  augment(frog_train) %>%
  rmse(latency, .pred)

# With the testing set

tree_fit %>%
  augment(frog_test) %>%
  rmse(latency, .pred)

## Use augment() and metrics() to compute a regression metric like mae()

tree_fit %>%
  augment(frog_train) %>%
  metrics(latency, .pred)

tree_fit %>%
  augment(frog_test) %>%
  metrics(latency, .pred)

#### REMEMBER THAT THE TESTING DATA IS PRECIOUS AND NEEDS TO BE HIDDEN. 
#### THEREFORE, WE NEED TO CREATIVELY MANIPULATE THE TRAINING DATA FOR VALIDATION

## Resampling and Cross-Validation

## Cross-Validation

vfold_cv(frog_train)

frog_folds <- vfold_cv(frog_train)
frog_folds$splits[1:3]

vfold_cv(frog_train, v = 5)

vfold_cv(frog_train, strata = latency)

set.seed(123)
frog_folds <- vfold_cv(frog_train, v = 10, strata = latency)
frog_folds

## Fit our model to the resamples

tree_res <- fit_resamples(tree_wflow, frog_folds)
tree_res

## Evaluating model performance

tree_res %>%
  collect_metrics()

## Comparing metrics
## How do the metrics from resampling compare to the metrics from training and testing?

tree_res %>%
  collect_metrics() %>% 
  select(.metric, mean, n)


# Remember that:

# the training set gives you overly optimistic metrics
# the test set is precious

## Evaluating model performance

# Save the assessment set results
ctrl_frog <- control_resamples(save_pred = TRUE)
tree_res <- fit_resamples(tree_wflow, frog_folds, control = ctrl_frog)

tree_preds <- collect_predictions(tree_res)
tree_preds


tree_preds %>% 
  ggplot(aes(latency, .pred, color = id)) + 
  geom_abline(lty = 2, col = "gray", size = 1.5) +
  geom_point(alpha = 0.5) +
  coord_obs_pred()

## Where are the fitted models?

tree_res

## Alternate resampling schemes

## Bootstrapping

set.seed(3214)
bootstraps(frog_train)

set.seed(322)
bootstraps(frog_train, times = 10)

## Validation set
## A validation set is just another type of resample

set.seed(853)
validation_split(frog_train, strata = latency)

## Random forest ###########################################

# Ensemble of many decision tree models
# All the trees vote!
# Bootstrap aggregating + random predictor sampling
# Often works well without tuning hyperparameters, as long as there are enough trees

## Create a random forest model

rf_spec <- rand_forest(trees = 1000, mode = "regression")
rf_spec

rf_wflow <- workflow(latency ~ ., rf_spec)
rf_wflow

## Evaluating model performance

ctrl_frog <- control_resamples(save_pred = TRUE)

# Random forest uses random numbers so set the seed first

set.seed(2)
rf_res <- fit_resamples(rf_wflow, frog_folds, control = ctrl_frog)
collect_metrics(rf_res)


collect_predictions(rf_res) %>% 
  ggplot(aes(latency, .pred, color = id)) + 
  geom_abline(lty = 2, col = "gray", size = 1.5) +
  geom_point(alpha = 0.5) +
  coord_obs_pred()


## Evaluate a workflow set
workflow_set(list(latency ~ .), list(tree_spec, rf_spec))


workflow_set(list(latency ~ .), list(tree_spec, rf_spec)) %>%
  workflow_map("fit_resamples", resamples = frog_folds) %>%
  rank_results()

## The first metric of the metric set is used for ranking. Use rank_metric to change that.
## Lots more available with workflow sets, like collect_metrics(), autoplot() methods, and more!


########### THE FINAL FIT ####################

# The final fit
# Suppose that we are happy with our random forest model.
# Let’s fit the model on the training set and verify our performance using the test set.
# We’ve shown you fit() and predict() (+ augment()) but there is a shortcut:

# frog_split has train + test info
final_fit <- last_fit(rf_wflow, frog_split) 

final_fit


collect_metrics(final_fit) # metrics with test set


collect_predictions(final_fit) # predictions with test set


collect_predictions(final_fit) %>%
  ggplot(aes(latency, .pred)) + 
  geom_abline(lty = 2, col = "deeppink4", size = 1.5) +
  geom_point(alpha = 0.5) +
  coord_obs_pred()

# Use this for prediction on new data, like for deploying

extract_workflow(final_fit)


# (5) Feature Engineering ######################################################

## Working with our predictors
## We might want to modify our predictors columns for a few reasons:
  
# The model requires them in a different format (e.g. dummy variables for lm()).
# The model needs certain data qualities (e.g. same units for K-NN).
# The outcome is better predicted when one or more columns are transformed 
# in some way (a.k.a “feature engineering”).
# The first two reasons are fairly predictable.
# The last one depends on your modeling problem.

##### look at tmwr.org Appendix for Preprocessing

## What is feature engineering?
## https://bookdown.org/max/FES/
## Think of a feature as some representation of a predictor that will be used in a model.

## Example representations:
  
# Interactions
# Polynomial expansions/splines
# PCA feature extraction

## Example: Dates

## How can we represent date columns for our model?
## When a date column is used in its native format, it is usually converted by an 
## R model to an integer.
## It can be re-engineered as:
  
# Days since a reference date
# Day of the week
# Month
# Year
# Indicators for holidays


## General definitions

# Data preprocessing steps allow your model to fit.
# Feature engineering steps help the model do the least work to predict the 
# outcome as well as possible.

# The recipes package can handle both!
# In a little bit, we’ll see successful (and unsuccessful) feature engineering 
# methods for our example data.

## Example Task
## The NHL data

## From Pittsburgh Penguins games, 7,471 shots / Data from the 2015-2016 season
## Let’s predict whether a shot is on-goal (a goal or blocked by goaltender) or not.

## Case study

library(tidymodels)
library(ongoal)

tidymodels_prefer()

glimpse(season_2015)


## Why a validation set?
## Recall that resampling gives us performance measures without using the test set.
## It’s important to get good resampling statistics (e.g. R2).
## That usually means having enough data to estimate performance.
## When you have “a lot” of data, a validation set can be an efficient way to do this.


## Splitting the NHL data

set.seed(23)
nhl_split <- initial_split(season_2015, prop = 3/4)
nhl_split

nhl_train_and_val <- training(nhl_split)
nhl_test  <- testing(nhl_split)

## not testing
nrow(nhl_train_and_val)

## testing
nrow(nhl_test)

## Validation split

## Since there are a lot of observations, we’ll use a validation set:
  
set.seed(234)
nhl_val <- validation_split(nhl_train_and_val, prop = 0.80)
nhl_val

## Remember that a validation split is a type of resample.

## Explore the training set data
nhl_train <- analysis(nhl_val$splits[[1]])

set.seed(100)
nhl_train %>% 
  sample_n(200) %>%
  plot_nhl_shots(emphasis = shooter_type)


## Prepare your data for modeling

# The recipes package is an extensible framework for pipeable sequences of 
# feature engineering steps that provide preprocessing tools to be applied to data.

# Statistical parameters for the steps can be estimated from an initial data set and
# then applied to other data sets.

# The resulting processed output can be used as inputs for statistical or machine 
# learning models.

## A first recipe

nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train)

## The recipe() function assigns columns to roles of 
## “outcome” or “predictor” using the formula.

summary(nhl_rec)

## Create indicator variables

## For any factor or character predictors, make binary indicators.
## There are many recipe steps that can convert categorical predictors to numeric columns.

nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_dummy(all_nominal_predictors())

## Filter out constant columns

nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())

## In case there is a factor level that was never observed in the training data 
## (resulting in a column of all 0s), we can delete any zero-variance predictors 
## that have a single unique value.

## Normalization

nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

## This centers and scales the numeric predictors.
## The recipe will use the training set to estimate the means and standard 
## deviations of the data.
## All data the recipe is applied to will be normalized using those statistics
## (there is no re-estimation).

## Reduce correlation

nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.9)

## To deal with highly correlated predictors, find the minimum set of predictor columns 
## that make the pairwise correlations less than the threshold.

## Other possible steps
## PCA feature extraction…
## step_pca(all_numeric_predictors())

## A fancy machine learning supervised dimension reduction technique…
## embed::step_umap(all_numeric_predictors(), outcome = on_goal)

## Nonlinear transforms like natural splines, and so on!
## step_ns(coord_y, coord_x, deg_free = 10)

## Create a recipe() for the on-goal data to :

# create one-hot indicator variables
# remove zero-variance variables

## Minimal recipe

nhl_indicators <-
  recipe(on_goal ~ ., data = nhl_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

## Using a workflow

set.seed(9)

nhl_glm_wflow <-
  workflow() %>%
  add_recipe(nhl_indicators) %>%
  add_model(logistic_reg())

ctrl <- control_resamples(save_pred = TRUE)
nhl_glm_res <-
  nhl_glm_wflow %>%
  fit_resamples(nhl_val, control = ctrl)

collect_metrics(nhl_glm_res)

## Holdout predictions

# Since we used `save_pred = TRUE`
glm_val_pred <- collect_predictions(nhl_glm_res)
glm_val_pred %>% slice(1:7)

## Two class data
## Let’s say we can define one class as the “event”, like a shot being on goal.

## The sensitivity is the true positive rate (accuracy on actual events).

## The specificity is the true negative rate (accuracy on actual non-events, 
## or 1 - false positive rate).

## These definitions assume that we know the threshold for converting 
## “soft” probability predictions into “hard” class predictions.

## Is a 50% threshold good?
  
## What happens if we say that we need to be 80% sure to declare an event?
  
## sensitivity ⬇️ and specificity ⬆️

## What happens for a 20% threshold?
  
## sensitivity ⬆️ and specificity ⬇️


## ROC curves

## To make an ROC (receiver operator characteristic) curve, we:
  
# calculate the sensitivity and specificity for all possible thresholds.
# plot false positive rate (x-axis) versus true positive rate (y-axis).

## We can use the area under the ROC curve as a classification metric:
  
# ROC AUC = 1 💯
# ROC AUC = 1/2 😢


# Assumes _first_ factor level is event; there are options to change that
roc_curve_points <- glm_val_pred %>% roc_curve(truth = on_goal, estimate = .pred_yes)
roc_curve_points %>% slice(1, 50, 100)


glm_val_pred %>% roc_auc(truth = on_goal, estimate = .pred_yes)


## ROC curve plot

autoplot(roc_curve_points)


## What do we do with the player data?

## There are 574 unique player values in our training set. How can we include
## this information in our model?
  
## We could:
  
# make the full set of indicator variables
# lump players who rarely shoot into an “other” group
# use feature hashing to create a smaller set of indicator variables
# use effect encoding to replace the shooter column with the estimated 
# effect of that predictor


## Collapsing factor levels

## There is a recipe step that will redefine factor levels based on the their 
## frequency in the training set:

nhl_other_rec <-
  recipe(on_goal ~ ., data = nhl_train) %>%
  # Any player with <= 0.01% of shots is set to "other"
  step_other(shooter, threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())


## Using this code, 362 players (out of 574) were collapsed into “other” based 
## on the training set. #########################################

## We could try to optimize the threshold for collapsing (see the next set of 
## slides on model tuning).######################################


## Does othering help?


nhl_other_wflow <-
  nhl_glm_wflow %>%
  update_recipe(nhl_other_rec)

nhl_other_res <-
  nhl_other_wflow %>%
  fit_resamples(nhl_val, control = ctrl)

collect_metrics(nhl_other_res)


## A little better ROC AUC and much faster to complete.

## Now let’s look at a more sophisticated tool called effect encodings.


## What is an effect encoding?

## We replace the qualitative’s predictor data with their effect on the outcome.

## Good statistical methods for estimating these rates use partial pooling.

## Pooling borrows strength across players and shrinks extreme values 
## (e.g. zero or one) towards the mean for players with very few shots.

## The embed package has recipe steps for effect encodings.


## Player effects

library(embed)

nhl_effect_rec <-
  recipe(on_goal ~ ., data = nhl_train) %>%
  step_lencode_mixed(shooter, goaltender, outcome = vars(on_goal)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

## It is very important to appropriately validate the effect encoding step 
## to make sure that we are not overfitting.


## Recipes are estimated

## Preprocessing steps in a recipe use the training set to compute quantities.
## What kind of quantities are computed for preprocessing?
  
# Levels of a factor
# Whether a column has zero variance
# Normalization
# Feature extraction
# Effect encodings

## When a recipe is part of a workflow, this estimation occurs when fit() is called.

## Effect encoding results
nhl_effect_wflow <-
  nhl_glm_wflow %>%
  update_recipe(nhl_effect_rec)

nhl_effect_res <-
  nhl_effect_wflow %>%
  fit_resamples(nhl_val, control = ctrl)

collect_metrics(nhl_effect_res)

## Increases AUC Better and it can handle new players (if they occur).######


## Angle

nhl_angle_rec <-
  nhl_effect_rec %>%
  step_mutate(
    angle = abs( atan2(abs(coord_y), (89 - coord_x) ) * (180 / pi) )
  )

## Shot from the defensive zone

nhl_zone_rec <-
  nhl_angle_rec %>%
  step_mutate(
    defensive_zone = ifelse(coord_x <= -25.5, 1, 0)
  )

## Behind goal line
nhl_behind_rec <-
  nhl_zone_rec %>%
  step_mutate(
    behind_goal_line = ifelse(coord_x >= 89, 1, 0)
  )


## Fit different recipes

## A workflow set can cross models and/or preprocessors and then resample them en masse.


no_coord_rec <- 
  nhl_indicators %>% 
  step_rm(starts_with("coord"))

set.seed(9)

nhl_glm_set_res <-
  workflow_set(
    list(`1_no_coord` = no_coord_rec,   `2_other` = nhl_other_rec, 
         `3_effects`  = nhl_effect_rec, `4_angle` = nhl_angle_rec, 
         `5_zone`     = nhl_zone_rec,   `6_bgl`   = nhl_behind_rec),
    list(logistic = logistic_reg())
  ) %>%
  workflow_map(fn = "fit_resamples", resamples = nhl_val, verbose = TRUE, control = ctrl)


## Compare recipes
library(forcats)
collect_metrics(nhl_glm_set_res) %>%
  filter(.metric == "roc_auc") %>%
  mutate(
    features = gsub("_logistic", "", wflow_id), 
    features = fct_reorder(features, mean)
  ) %>%
  ggplot(aes(x = mean, y = features)) +
  geom_point(size = 3) +
  labs(y = NULL, x = "ROC AUC (validation set)")


## Debugging a recipe

# Typically, you will want to use a workflow to estimate and apply a recipe.
# If you have an error and need to debug your recipe, the original recipe object 
# (e.g. encoded_players) can be estimated manually with a function called prep(). 
# It is analogous to fit().
# Another function (bake()) is analogous to predict(), and gives you the processed data back.
# The tidy() function can be used to get specific results from the recipe.


nhl_angle_fit <- prep(nhl_angle_rec)

tidy(nhl_angle_fit, number = 1) %>% slice(1:4)

bake(nhl_angle_fit, nhl_train %>% slice(1:3), starts_with("coord"), angle, shooter)

## More on recipes
# Once fit() is called on a workflow, changing the model does not re-fit the recipe.
# A list of all known steps is at https://www.tidymodels.org/find/recipes/.
# Some steps can be skipped when using predict().
# The order of the steps matters.

# (6) Tuning hyperparameters ######################################################




