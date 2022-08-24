# A gentle introduction to tidymodels
# https://rviews.rstudio.com/2019/06/19/a-gentle-intro-to-tidymodels/

## It is important to clarify that the group of packages that make up tidymodels do not 
## implement statistical models themselves. Instead, they focus on making all the tasks 
## around fitting the model much easier. Those tasks are data pre-processing 
## and results validation.

## The model has sub-steps
## Packages involved include:


### rsample - Different types of re-samples
### recipes - Transformations for model data pre-processing
### parnip - A common interface for model creation
### yardstick - Measure model performance

### Pre-process data (rsample and recipes)
### Train data (parsnip)
### Validate data (yardstick)

# (1) Example dataset is iris but this can be expanded to other models

# (2) Load the tidymodels library (contains a lot of support packages including tidyverse)

library(tidymodels)

# (3) Pre-process the data

## This step focuses on making data suitable for modeling by using data transformations.

# (4) Data Sampling

## The initial_split() function is specially built to separate 
## the data set into a training and testing set.
## prop is the train_test split

iris_split <- initial_split(iris, prop = 0.6)
iris_split

## Access the training data

iris_split %>% 
  training()

iris_split %>%
  training() %>%
  glimpse()

## Access the testing data

iris_split %>%
  testing()

iris_split %>%
  testing() %>%
  glimpse()

## The sampling functions are part of the rsample package

# (5) Pre-process interface

## recipe() - Starts a new set of transformations to be applied, 
## similar to the ggplot() command. Its main argument is the model’s formula.

## prep() - Executes the transformations on top of the data that is 
## supplied (typically, the training data).

## Each data transformation is a step. Functions correspond to specific 
## types of steps, each of which has a prefix of step_. There are several 
## step_ functions; in this example, we will use three of them:

### step_corr() - Removes variables that have large absolute correlations 
### with other variables

### step_center() - Normalizes numeric data to have a mean of zero

### step_scale() - Normalizes numeric data to have a standard deviation of one

## Another nice feature is that the step can be applied to a specific variable, 
## groups of variables, or all variables. The all_outcomes() and all_predictors() 
## functions provide a very convenient way to specify groups of variables. For 
## example, if we want the step_corr() to only analyze the predictor variables, 
## we use step_corr(all_predictors()). This capability saves us from having to 
## enumerate each variable.

## In this example (below), we use recipe() and prep() to create a recipe object.

iris_recipe <- training(iris_split) %>%
  recipe(Species ~.) %>%
  step_corr(all_predictors()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep()

## This function removed the Petal.Length, and centered and scales the other predictors.
## This was only applied to the training dataset

iris_recipe

# (6) Execute the pre-processing

## Once recipe has been prepped, it can be "baked()" with the testing() data.

iris_testing <- iris_recipe %>%
  bake(testing(iris_split)) 

glimpse(iris_testing)

## To load the prepared training data into a variable, we use juice(). 
## It will extract the data from the iris_recipe object.

iris_training <- juice(iris_recipe)

glimpse(iris_training)

# (7) Model Training

## Instead of replacing the modeling package, tidymodels replaces the interface. 
## Better said, tidymodels provides a single set of functions and arguments to 
## define a model. It then fits the model against the requested modeling package.

## In the example below, the rand_forest() function is used to initialize a Random 
## Forest model. To define the number of trees, the trees argument is used. To use 
## the ranger version of Random Forest, the set_engine() function is used. Finally, 
## to execute the model, the fit() function is used. The expected arguments are the 
## formula and data. Notice that the model runs on top of the juiced trained data.

iris_ranger <- rand_forest(trees = 100, mode = "classification") %>%
  set_engine("ranger") %>%
  fit(Species ~ ., data = iris_training)

## To compare models, simply change the modle type in set_engine()

iris_rf <-  rand_forest(trees = 100, mode = "classification") %>%
  set_engine("randomForest") %>%
  fit(Species ~ ., data = iris_training)

## The model definition is separated into smaller functions such as fit() and 
## set_engine(). This allows for a more flexible - and easier to learn - interface.

# (8) Predictions

predict(iris_ranger, iris_testing)

## Instead of a vector, the predict() function ran against a parsnip model returns
## a tibble. By default, the prediction variable is called .pred_class. In the 
## example, notice that the **baked testing data** is used.

## It is very easy to add the predictions to the baked testing data by 
## using dplyr’s bind_cols() function.

iris_ranger %>%
  predict(iris_testing) %>%
  bind_cols(iris_testing) %>%
  glimpse()

# (9) Model Validation

## Use the metrics() function to measure the performance of the model. 
## It will automatically choose metrics appropriate for a given type of model. 
## The function expects a tibble that contains the actual results (truth) and 
## what the model predicted (estimate).

iris_ranger %>%
  predict(iris_testing) %>%
  bind_cols(iris_testing) %>%
  metrics(truth = Species, estimate = .pred_class)

## Because of the consistency of the new interface, measuring the same metrics 
## against the randomForest model is as easy as replacing the model variable 
## at the top of the code.

iris_rf %>%
  predict(iris_testing) %>%
  bind_cols(iris_testing) %>%
  metrics(truth = Species, estimate = .pred_class)

# (10) Per classifier metrics

## It is easy to obtain the probability for each possible predicted value by setting 
## the type argument to prob. That will return a tibble with as many variables as 
## there are possible predicted values. Their name will default to the original 
## value name, prefixed with .pred_.

iris_ranger %>%
  predict(iris_testing, type = "prob") %>%
  glimpse()

## Again, use bind_cols() to append the predictions to the baked testing data set.

iris_probs <- iris_ranger %>%
  predict(iris_testing, type = "prob") %>%
  bind_cols(iris_testing)

glimpse(iris_probs)

## Now that everything is in one tibble, it is easy to calculate curve methods. 
## In this case we are using gain_curve().

iris_probs%>%
  gain_curve(Species, .pred_setosa:.pred_virginica) %>%
  glimpse()

## The curve methods include an autoplot() function that easily creates 
## a ggplot2 visualization.

iris_probs%>%
  gain_curve(Species, .pred_setosa:.pred_virginica) %>%
  autoplot()

## This is an example of a roc_curve(). Again, because of the consistency 
## of the interface, only the function name needs to be modified; even 
## the argument values remain the same.

iris_probs%>%
  roc_curve(Species, .pred_setosa:.pred_virginica) %>%
  autoplot()

## To measured the combined single predicted value and the probability of each 
## possible value, combine the two prediction modes (with and without prob type). 
## In this example, using dplyr’s select() makes the resulting tibble easier to read.

predict(iris_ranger, iris_testing, type = "prob") %>%
  bind_cols(predict(iris_ranger, iris_testing)) %>%
  bind_cols(select(iris_testing, Species)) %>%
  glimpse()

## Pipe the resulting table into metrics(). In this case, 
## specify .pred_class as the estimate.

predict(iris_ranger, iris_testing, type = "prob") %>%
  bind_cols(predict(iris_ranger, iris_testing)) %>%
  bind_cols(select(iris_testing, Species)) %>%
  metrics(Species, .pred_setosa:.pred_virginica, estimate = .pred_class)

