# https://www.tmwr.org/
# Tools for Creating Effective Models

# (A) Resampling for Evaluating Performance

# (10.1) The Resubstitution Approach

## When we measure performance on the same data that we used for training 
## (as opposed to new data or testing data), we say we have resubstituted the data

## Use Ames data. It includes a recipe object named ames_rec, a linear model, and a 
## workflow using that recipe and model called lm_wflow. This workflow was fit on 
## the training set, resulting in lm_fit.

## Random forests are a tree ensemble method that operates by creating a large number 
## of decision trees from slightly different versions of the training set 

## Random forest models are very powerful, and they can emulate the underlying data 
## patterns very closely. While this model can be computationally intensive, it is very 
## low maintenance; very little preprocessing is required

## Using the same predictor set as the linear model (without the extra preprocessing steps), 
## we can fit a random forest model to the training set via the "ranger" engine (which uses 
## the ranger R package for computation). This model requires no preprocessing, so a simple 
## formula can be used:

rf_model <- 
  rand_forest(trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wflow <- 
  workflow() %>% 
  add_formula(
    Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
      Latitude + Longitude) %>% 
  add_model(rf_model) 

rf_fit <- rf_wflow %>% fit(data = ames_train)

## How should we compare the linear and random forest models? For demonstration, we will
## predict the training set to produce what is known as an apparent metric or resubstitution 
## metric. This function creates predictions and formats the results:

estimate_perf <- function(model, dat) {
  # Capture the names of the `model` and `dat` objects
  cl <- match.call()
  obj_name <- as.character(cl$model)
  data_name <- as.character(cl$dat)
  data_name <- gsub("ames_", "", data_name)
  
  # Estimate these metrics:
  reg_metrics <- metric_set(rmse, rsq)
  
  model %>%
    predict(dat) %>%
    bind_cols(dat %>% select(Sale_Price)) %>%
    reg_metrics(Sale_Price, .pred) %>%
    select(-.estimator) %>%
    mutate(object = obj_name, data = data_name)
}


## Both RMSE and Rsquared are computed. The resubstitution statistics are:

estimate_perf(rf_fit, ames_train)

estimate_perf(lm_fit, ames_train)

## Based on these results, the random forest is much more capable of predicting the 
## sale prices; the RMSE estimate is two-fold better than linear regression. If we 
## needed to choose between these two models for this price prediction problem, 
## we would probably chose the random forest because, on the log scale we are using, 
## its RMSE is about half as large. The next step applies the random forest model to 
## the test set for final verification:

estimate_perf(rf_fit, ames_test)

## In this context, bias is the difference between the true pattern or relationships 
## in data and the types of patterns that the model can emulate. Many black-box machine 
## learning models have low bias, meaning they can reproduce complex relationships. 
## Other models (such as linear/logistic regression, discriminant analysis, and others) 
## are not as adaptable and are considered high bias models.

## (10.2) Resampling Methods

## Resampling methods are empirical simulation systems that emulate the process of using 
## some data for modeling and different data for evaluation. Most resampling methods are 
## iterative, meaning that this process is repeated multiple times.


