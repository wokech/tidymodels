# https://www.tmwr.org/
# Modeling Basics
# (A) The Ames Housing Data

## The data set contains information on 2,930 properties in Ames, Iowa, 
## including columns related to:

## house characteristics (bedrooms, garage, fireplace, pool, porch, etc.)
## location (neighborhood)
## lot information (zoning, shape, size, etc.)
## ratings of condition and quality
## sale price

## Our modeling goal is to predict the sale price of a house based on other information we have, 
## such as its characteristics and location.

# (1) Load the required libraries and data

library(tidyverse)
library(tidymodels)
library(modeldata)

data(ames)

str(ames)
dim(ames)
glimpse(ames)

# (2) EDA

## Exploratory data analysis by focusing on the outcome we want 
## to predict: the last sale price of the house (in USD). We can 
## create a histogram to see the distribution of sale prices. 

tidymodels_prefer() # in case of conflicts with other packages

ggplot(ames, aes(x = Sale_Price)) + 
  geom_histogram(bins = 50, col= "white")

## Log-transform the data

## When modeling this outcome, a strong argument can be made that the price should 
## be log-transformed. The advantages of this type of transformation are that no 
## houses would be predicted with negative sale prices and that errors in predicting 
## expensive houses will not have an undue influence on the model.

## Also, from a statistical perspective, a logarithmic transform may also stabilize 
## the variance in a way that makes inference more legitimate.

ggplot(ames, aes(x = Sale_Price)) + 
  geom_histogram(bins = 50, col= "white") +
  scale_x_log10()

## The disadvantages of transforming the outcome mostly 
## relate to interpretation of model results.

## From this point on, the outcome column is prelogged in the ames data frame.

ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))

##  Some basic questions that could be examined during this exploratory stage include:

## Is there anything odd or noticeable about the distributions of the 
## individual predictors? Is there much skewness or any pathological distributions?
  
## Are there high correlations between predictors? For example, there are 
## multiple predictors related to house size. Are some redundant?
  
## Are there associations between predictors and the outcomes?

# (B) Spending our Data

## There are several steps to creating a useful model, including 
## parameter estimation, model selection and tuning, and performance assessment.

## How should the data be applied to different steps or tasks? The idea of 
## data spending is an important first consideration when modeling, especially 
## as it relates to empirical validation.

## When data are reused for multiple tasks, instead of carefully “spent” from 
## the finite data budget, certain risks increase, such as the risk of accentuating 
## bias or compounding effects from methodological errors.

## When there are copious amounts of data available, a smart strategy is to 
## allocate specific subsets of data for different tasks, as opposed to allocating 
## the largest possible amount (or even all) to the model parameter estimation only.

## This chapter demonstrates the basics of splitting (i.e., creating a data budget) 
## for our initial pool of samples for different purposes.

# (1) Common methods for splitting data

## The primary approach for empirical model validation is to split the existing pool 
## of data into two distinct sets, the training set and the test set.

## One portion of the data is used to develop and optimize the model. 
## This training set is usually the majority of the data.

## The other portion of the data is placed into the test set. This is held in 
## reserve until one or two models are chosen as the methods most likely to succeed.

## How should we conduct this split of the data? The answer depends on the context.

## The rsample package has tools for making data splits such as this; the function 
## initial_split() was created for this purpose.


## Set the random number stream using `set.seed()` so that the results can be 
## reproduced later. 
set.seed(501)

## Save the split information for an 80/20 split of the data
ames_split <- initial_split(ames, prop = 0.80)
ames_split

## The object ames_split is an rsplit object and contains only the partitioning 
## information; to get the resulting data sets, we apply two more functions:

ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

dim(ames_train)
dim(ames_test)

## Simple random sampling is appropriate in many cases but there are exceptions. 
## When there is a dramatic class imbalance in classification problems, one class 
## occurs much less frequently than another. Using a simple random sample may 
## haphazardly allocate these infrequent samples disproportionately into the 
## training or test set. To avoid this, stratified sampling can be used. 
## The training/test split is conducted separately within each class and then these 
## subsamples are combined into the overall training and test set. 
## For regression problems, the outcome data can be artificially binned into quartiles 
## and then stratified sampling can be conducted four separate times. 
## This is an effective method for keeping the distributions of the outcome 
## similar between the training and test set

## With skewed data, simple splitting is not ideal. 
## You need to use stratified random sampling.

## Only a single column can be used for stratification.

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

dim(ames_train)
dim(ames_test)

## For time series data, the rsample package contains a function called 
## initial_time_split() that is very similar to initial_split(). 

## The proportion of data that should be allocated for splitting is highly 
## dependent on the context of the problem at hand. Too little data in the 
## training set hampers the model’s ability to find appropriate parameter estimates. 
## Conversely, too little data in the test set lowers the quality of the performance 
## estimates. Parts of the statistics community eschew test sets in general because 
## they believe all of the data should be used for parameter estimation.

# (2) validation Data

## Overfit means that the model performed very well on the 
## training set but poorly on the test set. To combat this issue, a small 
## validation set of data were held back and used to measure performance as 
## the network was trained. Once the validation set error rate began to rise, 
## the training would be halted. In other words, the validation set was a means 
## to get a rough sense of how well the model performed prior to the test set.

# (3) Multilevel Data

## With the Ames housing data, a property is considered to be 
## the independent experimental unit.

# (4) Other considerations for a data budget

## The problem of information leakage occurs when data outside of the training set 
## are used in the modeling process.

## Keeping the training data in a separate data frame from the test set is one small 
## check to make sure that information leakage does not occur by accident.

## Second, techniques to subsample the training set can mitigate specific issues 
## (e.g., class imbalances). This is a valid and common technique that deliberately 
## results in the training set data diverging from the population from which the 
## data were drawn.

## Data splitting is the fundamental strategy for empirical validation of models. 
## Even in the era of unrestrained data collection, a typical modeling project has 
## a limited amount of appropriate data, and wise spending of a project’s 
## data is necessary.

# (C) Fitting models with parsnip

## The parsnip package, one of the R packages that are part of the tidymodels metapackage, 
## provides a fluent and standardized interface for a variety of different models.

## Specifically, we will focus on how to fit() and predict() directly with a parsnip object, 
## which may be a good fit for some straightforward modeling problems. 

# (1) Create a model

## Once the data have been encoded in a format ready for a modeling algorithm, such as a 
## numeric matrix, they can be used in the model building process.

## Linear regression

## Methods
## Ordinary linear regression uses the traditional method of least squares 
## to solve for the model parameters.
## Regularized linear regression adds a penalty to the least squares method 
## to encourage simplicity by removing predictors and/or shrinking their 
## coefficients towards zero. This can be executed using Bayesian or 
## non-Bayesian techniques.

## The common ways to use ordinary and regularized linear regression

## Ordinary: model <- lm(formula, data, ...)
## stats package

## Regularized/Bayesian: model <- stan_glm(formula, data, family = "gaussian", ...)
## rstanarm package

## Regularized/Non-Bayesian: model <- glmnet(x = matrix, y = vector, family = "gaussian", ...)
## glmnet package

## Tidymodels allows one to specify a model using:
## 1) type of model (lin reg, KNN, rand forest,...)
## 2) engine for fitting the model (stan, glmnet,....)
## 3) mode of the model (outcomes like classification or regression)


## Specify the details of the model

library(tidymodels)
tidymodels_prefer()

linear_reg() %>% set_engine("lm")

linear_reg() %>% set_engine("glmnet") 

linear_reg() %>% set_engine("stan")

## Once the details of the model have been specified, the model estimation can be done
## with either the fit() function (to use a formula) or the fit_xy() function 
## (when your data are already pre-processed).

## The translate() function can provide details on how parsnip converts the user’s code 
## to the package’s syntax.

linear_reg() %>% set_engine("lm") %>% translate()

linear_reg(penalty = 1) %>% set_engine("glmnet") %>% translate()

linear_reg() %>% set_engine("stan") %>% translate()

## Note that missing_arg() is just a placeholder for the data that 
## has yet to be provided.

## We supplied a required penalty argument for the glmnet engine. Also, for the 
## Stan and glmnet engines, the family argument was automatically added as a 
## default. As will be shown later in this section, this option can be changed.

## Let’s walk through how to predict the sale price of houses in the 
## Ames data as a function of only longitude and latitude.

lm_model <- 
linear_reg() %>% 
  set_engine("lm")

lm_form_fit <- 
  lm_model %>% 
  # Recall that Sale_Price has been pre-logged
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

lm_xy_fit <- 
  lm_model %>% 
  fit_xy(
    x = ames_train %>% select(Longitude, Latitude),
    y = ames_train %>% pull(Sale_Price)
  )

lm_form_fit

lm_xy_fit

## *** parsnip *** allows us to have consistent argument names

## Example

rand_forest(trees = 1000, min_n = 5) %>% 
  set_engine("ranger") %>% 
  set_mode("regression") %>% 
  translate()

rand_forest(trees = 1000, min_n = 5) %>% 
  set_engine("ranger", verbose = TRUE) %>% 
  set_mode("regression") 

# (2) Use the model results

## Several quantities are stored in a parsnip model object, including the fitted model. 
## This can be found in an element called fit, which can be returned using 
## the extract_fit_engine() function.

lm_form_fit %>% extract_fit_engine()

lm_form_fit %>% extract_fit_engine() %>% vcov()

## Never pass the fit element of a parsnip model to a model prediction function, 
## i.e., use predict(lm_form_fit) but do not use predict(lm_form_fit$fit)

## summary() method for lm objects can be used to print the results of the 
## model fit, including a table with parameter values, their uncertainty 
## estimates, and p-values.

model_res <- 
  lm_form_fit %>% 
  extract_fit_engine() %>% 
  summary()

## The model coefficient table is accessible via the `coef` method.
param_est <- coef(model_res)
class(param_est)

param_est

## The broom package can convert many types of model objects to a tidy structure. 
## For example, using the tidy() method on the linear model produces:

tidy(lm_form_fit)

# (3) Make predictions

## For example, when numeric data are predicted

ames_test_small <- ames_test %>% slice(1:5)
predict(lm_form_fit, new_data = ames_test_small)

## The row order of the predictions are always the same as the original data.

## IMPORTANT!!!!!:
## Why the leading dot in some of the column names? Some tidyverse and tidymodels 
## arguments and return values contain periods. This is to protect against merging 
## data with duplicate names. There are some data sets that contain predictors 
## named pred!

## Merge predictions with the original data

ames_test_small %>% 
  select(Sale_Price) %>% 
  bind_cols(predict(lm_form_fit, ames_test_small)) %>% 
  # Add 95% prediction intervals to the results:
  bind_cols(predict(lm_form_fit, ames_test_small, type = "pred_int")) 

## Suppose that we used a decision tree to model the Ames data. 
## Outside of the model specification, there are no significant 
## differences in the code pipeline

tree_model <- 
  decision_tree(min_n = 2) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

tree_fit <- 
  tree_model %>% 
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

ames_test_small %>% 
  select(Sale_Price) %>% 
  bind_cols(predict(tree_fit, ames_test_small))

# (4) Parsnip-extension packages

## https://www.tidymodels.org/find/

# (5) Creating model specifications

## Use this code: parsnip_addin()

# (D) A Model Workflow

## This chapter introduces a new concept called a model workflow. 
## The purpose of this concept (and the corresponding tidymodels 
## workflow() object) is to encapsulate the major pieces of the modeling process.

# (1) Where does the model begin and end?

## So far, when we have used the term “the model,” we have meant a 
## equation that relates some predictors to one or more outcomes.

## The conventional way of thinking about the modeling process is 
## that it only includes the model fit.

## For some straightforward data sets, fitting the model itself may be the 
## entire process. However, a variety of choices and additional steps often 
## occur before the model is fit:

## While our example model has p predictors, it is common to start with more than  
## p candidate predictors. Through exploratory data analysis or using domain 
## knowledge, some of the predictors may be excluded from the analysis. 
## In other cases, a feature selection algorithm may be used to make a data-driven 
## choice for the minimum predictor set for the model.

## There are times when the value of an important predictor is missing. Rather than 
## eliminating this sample from the data set, the missing value could be imputed 
## using other values in the data.

## It may be beneficial to transform the scale of a predictor. 
## We can estimate the proper scale using a statistical transformation technique, 
## the existing data, and some optimization criterion. Other transformations, 
## such as PCA, take groups of predictors and transform them into new features 
## that are used as the predictors.

## It is important to focus on the broader modeling process, instead of only fitting 
## the specific model used to estimate parameters. This broader process includes any 
## preprocessing steps, the model fit itself, as well as potential post-processing 
## activities.

## We call the sequence of computational operations related to modeling workflows.

## Example

## PCA is a way to replace correlated predictors with new artificial features that 
## are uncorrelated and capture most of the information in the original set. 
## The new features could be used as the predictors, and least squares regression 
## could be used to estimate the model parameters.

# (2) Workflow Basics

## The workflows package allows the user to bind modeling and preprocessing objects 
## together. Let’s start again with the Ames data and a simple linear model.

library(tidymodels)  # Includes the workflows package
tidymodels_prefer()

lm_model <- 
  linear_reg() %>% 
  set_engine("lm")

lm_wflow <- 
  workflow() %>% 
  add_model(lm_model)

lm_wflow

## Use the standard R formula as a preprocessor

lm_wflow <- 
  lm_wflow %>% 
  add_formula(Sale_Price ~ Longitude + Latitude)

lm_wflow

## Workflows have a fit() method that can be used to create the model. 

lm_fit <- fit(lm_wflow, ames_train)
lm_fit

## We can also predict() on the fitted workflow.

predict(lm_fit, ames_test %>% slice(1:5))

## Both the model and preprocessor can be removed or updated.

lm_fit %>% update_formula(Sale_Price ~ Longitude)

# (3) Adding raw variables to the workflow()

## There is another interface for passing data to the model, 
## the add_variables() function, which uses a dplyr-like syntax for 
## choosing variables. 
## The function has two primary arguments: outcomes and predictors.

lm_wflow <- 
  lm_wflow %>% 
  remove_formula() %>% 
  add_variables(outcome = Sale_Price, predictors = c(Longitude, Latitude))

lm_wflow

### review ERRORS!!###########################

## The predictors could also have been specified using a 
## more general selector, such as:

predictors = c(ends_with("tude"))

## One nicety is that any outcome columns accidentally specified in the 
## predictors argument will be quietly removed.

predictors = everything()

## When the model is fit, the specification assembles these data, unaltered, 
## into a data frame and passes it to the underlying function:

#(4) How does a workflow() use the formula?

## A workflow is a general purpose interface. When add_formula() is used, how 
## should the workflow preprocess the data? Since the preprocessing is model 
## dependent, workflows attempts to emulate what the underlying model would do 
## whenever possible. If it is not possible, the formula processing should not 
## do anything to the columns used in the formula.

# Tree-based Models

# (4.1) Special formulas and inline functions

##  To fit a regression model that has random effects for subjects.
install.packages("lme4")
library(nlme)
library(lme4)
lmer(distance ~ Sex + (age | Subject), data = Orthodont)

## The problem is that standard R methods can’t properly process this formula:

model.matrix(distance ~ Sex + (age | Subject), data = Orthodont)

## The issue is that the special formula has to be processed by the underlying 
## package code, not the standard model.matrix() approach.

tidymodels_prefer()
install.packages("multilevelmod")
library(multilevelmod)
library(rstanarm)

multilevel_spec <- linear_reg() %>% set_engine("lmer")

multilevel_workflow <- 
  workflow() %>% 
  # Pass the data along as-is: 
  add_variables(outcome = distance, predictors = c(Sex, age, Subject)) %>% 
  add_model(multilevel_spec, 
            # This formula is given to the model
            formula = distance ~ Sex + (age | Subject))

multilevel_fit <- fit(multilevel_workflow, data = Orthodont)
multilevel_fit

library(survival)
install.packages("censored")
library(censored)

parametric_spec <- survival_reg()

parametric_workflow <- 
  workflow() %>% 
  add_variables(outcome = c(fustat, futime), predictors = c(age, rx)) %>% 
  add_model(parametric_spec, 
            formula = Surv(futime, fustat) ~ age + strata(rx))

parametric_fit <- fit(parametric_workflow, data = ovarian)
parametric_fit

# (5) Creating multiple workflows at once

## For predictive models, it is advisable to evaluate a variety of different model types. 
## This requires the user to create multiple model specifications.

## Sequential testing of models typically starts with an expanded set of predictors. 
## This “full model” is compared to a sequence of the same model that removes each 
## predictor in turn. Using basic hypothesis testing methods or empirical validation, 
## the effect of each predictor can be isolated and assessed.


## As an example, let’s say that we want to focus on the different ways that house location 
## is represented in the Ames data. We can create a set of formulas that capture these predictors:

location <- list(
  longitude = Sale_Price ~ Longitude,
  latitude = Sale_Price ~ Latitude,
  coords = Sale_Price ~ Longitude + Latitude,
  neighborhood = Sale_Price ~ Neighborhood
)

library(workflowsets)
location_models <- workflow_set(preproc = location, models = list(lm = lm_model))
location_models

location_models$info[[1]]

extract_workflow(location_models, id = "coords_lm")


## In the meantime, let’s create model fits for each formula and save them in a new column 
## called fit. We’ll use basic dplyr and purrr operations:

location_models <-
  location_models %>%
  mutate(fit = map(info, ~ fit(.x$workflow[[1]], ames_train)))
location_models

location_models$fit[[1]]

## In general, there’s a lot more to workflow sets! While we’ve covered the basics here, 
## the nuances and advantages of workflow sets.

# (6) Evaluating the Test Set

## Let’s say that we’ve concluded our model development and have settled on a final model. 
## There is a convenience function called last_fit() that will fit the model to the 
## entire training set and evaluate it with the testing set.

final_lm_res <- last_fit(lm_wflow, ames_split)
final_lm_res

fitted_lm_wflow <- extract_workflow(final_lm_res)

collect_metrics(final_lm_res)
collect_predictions(final_lm_res) %>% slice(1:5)

# (E) Feature Engineering with recipes

## Feature engineering entails reformatting predictor values to make them 
## easier for a model to use effectively. 

## This includes transformations and encodings of the data to best represent 
## their important characteristics.

## Imagine that you have two predictors in a data set that can be more effectively 
## represented in your model as a ratio; creating a new predictor from the ratio 
## of the original two is a simple example of feature engineering.

## Take the location of a house in Ames as a more involved example. 
## There are a variety of ways that this spatial information can be exposed to a model, 
## including neighborhood (a qualitative measure), longitude/latitude, distance to the 
## nearest school or Iowa State University, and so on.

## The original format of the data, for example numeric (e.g., distance) versus 
## categorical (e.g., neighborhood), is also a driving factor in feature engineering choices.

## IMPORTANT
## Other examples of preprocessing to build better features for modeling include:

  ## Correlation between predictors can be reduced via feature extraction or the removal of 
  ## some predictors.

  ## When some predictors have missing values, they can be imputed using a sub-model.

  ## Models that use variance-type measures may benefit from coercing the distribution of 
  ## some skewed predictors to be symmetric by estimating a transformation.

## Feature engineering and data preprocessing can also involve reformatting that may be required 
## by the model. Some models use geometric distance metrics and, consequently, numeric predictors 
## should be centered and scaled so that they are all in the same units. Otherwise, the distance 
## values would be biased by the scale of each column.

## IMPORTANT ##
## Different models have different preprocessing requirements and some, such as tree-based models, 
## require very little preprocessing at all.

## In this chapter, we introduce the recipes package that you can use to combine different feature 
## engineering and preprocessing tasks into a single object and then apply these transformations 
## to different data sets.

# (1) A simple recipe() for the Ames Housing Data

## In this section, we will focus on a small subset of the predictors 
## available in the Ames housing data:

  ## The neighborhood (qualitative, with 29 neighborhoods in the training set)

  ## The gross above-grade living area (continuous, named Gr_Liv_Area)

  ## The year built (Year_Built)

  ## The type of building (Bldg_Type)

## Suppose that an initial ordinary linear regression model were fit to these data. 

lm(Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + Year_Built + Bldg_Type, data = ames)

## When this function is executed, the data are converted from a data frame to a numeric 
## design matrix (also called a model matrix) and then the least squares method is used 
## to estimate parameters.

## What this formula does can be decomposed into a series of steps:

  ## Sale price is defined as the outcome while neighborhood, gross living area, the year built, 
  ## and building type variables are all defined as predictors.

  ## A log transformation is applied to the gross living area predictor.

  ## The neighborhood and building type columns are converted from a non-numeric format to a 
  ## numeric format (since least squares requires numeric predictors).

##  The formula method will apply these data manipulations to any data, including new data, 
## that are passed to the predict() function.


## A recipe is also an object that defines a series of steps for data processing. Unlike the 
## formula method inside a modeling function, the recipe defines the steps via step_*() functions 
## without immediately executing them; it is only a specification of what should be done.


library(tidymodels) # Includes the recipes package
tidymodels_prefer()

simple_ames <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_dummy(all_nominal_predictors())
simple_ames


## Let’s break this down:

  ## The call to recipe() with a formula tells the recipe the roles of the “ingredients” or 
  ## variables (e.g., predictor, outcome). It only uses the data ames_train to determine the 
  ## data types for the columns.

  ## step_log() declares that Gr_Liv_Area should be log transformed.

  ## step_dummy() specifies which variables should be converted from a qualitative format 
  ## to a quantitative format, in this case, using dummy or indicator variables. An indicator 
  ## or dummy variable is a binary numeric variable (a column of ones and zeroes) that 
  ## encodes qualitative information; we will dig deeper into these kinds of 
  ## variables in Section 8.4.1.

## IMPORTANT

  ## Other selectors specific to the recipes package are: all_numeric_predictors(), all_numeric(), 
  ## all_predictors(), and all_outcomes(). As with dplyr, one or more unquoted expressions, 
  ## separated by commas, can be used to select which columns are affected by each step.


## What is the advantage to using a recipe, over a formula or raw predictors? 
## There are a few, including:

  ## These computations can be recycled across models since they are 
  ## not tightly coupled to the modeling function.
  
  ## A recipe enables a broader set of data processing choices than formulas can offer.
  
  ## The syntax can be very compact. For example, all_nominal_predictors() can be used 
  ## to capture many variables for specific types of processing while a formula would 
  ## require each to be explicitly listed.

  ## All data processing can be captured in a single R object instead of in scripts that 
  ## are repeated, or even spread across different files.

  ## The function all_nominal_predictors() captures the names of any predictor columns 
  ## that are currently factor or character (i.e., nominal) in nature. This is a dplyr-like 
  ## selector function similar to starts_with() or matches() but that can only be used inside 
  ## of a recipe.

# (2) Using recipes

## Preprocessing choices and feature engineering should typically be considered part of a 
## modeling workflow, not a separate task.

## We can have only one preprocessing method at a time, so we need to remove the existing 
## preprocessor before adding the recipe.

lm_wflow <- 
  lm_wflow %>% 
  remove_variables() %>% 
  add_recipe(simple_ames)
lm_wflow

## Let’s estimate both the recipe and model using a simple call to fit():

lm_fit <- fit(lm_wflow, ames_train)

## The predict() method applies the same preprocessing that was used on the 
## training set to the new data before passing them along to the model’s predict() method

predict(lm_fit, ames_test %>% slice(1:3))

## If we need the bare model object or recipe, there are extract_* functions 
## that can retrieve them

lm_fit %>% 
  extract_recipe(estimated = TRUE)


# To tidy the model fit: 
lm_fit %>% 
  # This returns the parsnip object:
  extract_fit_parsnip() %>% 
  # Now tidy the linear model object:
  tidy() %>% 
  slice(1:5)

# (3) How data are used by the recipe()

## Data are passed to recipes at different stages.

## First, when calling recipe(..., data), the data set is used to determine the data types of 
## each column so that selectors such as all_numeric() or all_numeric_predictors() can be used.

## Second, when preparing the data using fit(workflow, data), the training data are used for 
## all estimation operations including a recipe that may be part of the workflow, from 
## determining factor levels to computing PCA components and everything in between.

##### IMPORTANT #####
## All preprocessing and feature engineering steps use only the training data. Otherwise, 
## information leakage can negatively impact the model’s performance when used with new data.
#####

## Finally, when using predict(workflow, new_data), no model or preprocessor parameters 
## like those from recipes are re-estimated using the values in new_data. Take centering and 
## scaling using step_normalize() as an example. Using this step, the means and standard 
## deviations from the appropriate columns are determined from the training set; new samples 
## at prediction time are standardized using these values from training when predict() is invoked.

# (4) Example of recipe steps

## Before proceeding, let’s take an extended tour of the capabilities of recipes and explore 
## some of the most important step_*() functions. These recipe step functions each specify a 
## specific possible step in a feature engineering process, and different recipe steps can 
## have different effects on columns of data.

# (4.1) Encoding qualitative data in a numeric format

## One of the most common feature engineering tasks is transforming nominal or qualitative 
## data (factors or characters) so that they can be encoded or represented numerically. 
## Sometimes we can alter the factor levels of a qualitative column in helpful ways prior 
## to such a transformation. For example, step_unknown() can be used to change missing values 
## to a dedicated factor level. Similarly, if we anticipate that a new factor level may be 
## encountered in future data, step_novel() can allot a new level for this purpose.

## Additionally, step_other() can be used to analyze the frequencies of the factor levels in 
## the training set and convert infrequently occurring values to a catch-all level of “other,” 
## with a threshold that can be specified. A good example is the Neighborhood predictor
## in our data.

## Here we see that two neighborhoods have less than five properties in the training 
## data (Landmark and Green Hills); in this case, no houses at all in the Landmark 
## neighborhood were included in the training set.

## If we add step_other(Neighborhood, threshold = 0.01) to our recipe, the bottom 1% of 
## the neighborhoods will be lumped into a new level called “other.”

simple_ames <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors())

########
## Many, but not all, underlying model calculations require predictor values to be encoded 
## as numbers. Notable exceptions include tree-based models, rule-based models, and naive 
## Bayes models.
########

## The most common method for converting a factor predictor to a numeric format is to 
## create dummy or indicator variables.  For dummy variables, the single Bldg_Type column 
## would be replaced with four numeric columns whose values are either zero or one. 

## The full set of encodings can be used for some models. This is traditionally called the 
## one-hot encoding and can be achieved using the one_hot argument of step_dummy(). One helpful 
## feature of step_dummy() is that there is more control over how the resulting dummy variables 
## are named.

## Traditional dummy variables require that all of the possible categories be known to create 
## a full set of numeric features. There are other methods for doing this transformation to a 
## numeric format. Feature hashing methods only consider the value of the category to assign 
## it to a predefined pool of dummy variables. Effect or likelihood encodings replace the 
## original data with a single numeric column that measures the effect of those data. Both 
## feature hashing and effect encoding can seamlessly handle situations where a novel factor 
## level is encountered in the data

########
## Different recipe steps behave differently when applied to variables in the data. For example, 
## step_log() modifies a column in place without changing the name. Other steps, such as 
## step_dummy(), eliminate the original data column and replace it with one or more columns with 
## different names. The effect of a recipe step depends on the type of feature engineering 
## transformation being done.
########

# (4.2) Interaction terms

## Interaction effects involve two or more predictors. Such an effect occurs when one predictor 
## has an effect on the outcome that is contingent on one or more other predictors.

## If the relationship between a predictor and outcome changes as a function of 
## another predictor, then we can add an interaction term between the two predictors 
## to the model along with the original two predictors (which are called the main effects).

## After exploring the Ames training set, we might find that the regression slopes for 
## the gross living area differ for different building types.

ggplot(ames_train, aes(x = Gr_Liv_Area, y = 10^Sale_Price)) + 
  geom_point(alpha = .2) + 
  facet_wrap(~ Bldg_Type) + 
  geom_smooth(method = lm, formula = y ~ x, se = FALSE, color = "lightblue") + 
  scale_x_log10() + 
  scale_y_log10() + 
  labs(x = "Gross Living Area", y = "Sale Price (USD)")

## How are interactions specified in a recipe? 

## A base R formula would take an interaction 
## using a :, so we would use:

Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + Bldg_Type + 
  log10(Gr_Liv_Area):Bldg_Type
# or
Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) * Bldg_Type 

## where * expands those columns to the main effects and interaction term.

## Recipes are more explicit and sequential, and they give you more control. With the 
## current recipe, step_dummy() has already created dummy variables. How would we combine
## these for an interaction? The additional step would look like 
## step_interact(~ interaction terms) where the terms on the right-hand side of the tilde 
## are the interactions. 

simple_ames <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  # Gr_Liv_Area is on the log scale from a previous step
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_"))

# (4.3) Spline functions

## When a predictor has a nonlinear relationship with the outcome, some types of predictive 
## models can adaptively approximate this relationship during training. However, simpler is 
## usually better and it is not uncommon to try to use a simple model, such as a linear fit, 
## and add in specific nonlinear features for predictors that may need them, such as longitude 
## and latitude for the Ames housing data.

## One common method for doing this is to use spline functions to represent the data. Splines 
## replace the existing numeric predictor with a set of columns that allow a model to emulate 
## a flexible, nonlinear relationship.

## As more spline terms are added to the data, the capacity to nonlinearly represent the 
## relationship increases. Unfortunately, it may also increase the likelihood of picking up 
## on data trends that occur by chance (i.e., overfitting).

library(patchwork)
library(splines)

plot_smoother <- function(deg_free) {
  ggplot(ames_train, aes(x = Latitude, y = 10^Sale_Price)) + 
    geom_point(alpha = .2) + 
    scale_y_log10() +
    geom_smooth(
      method = lm,
      formula = y ~ ns(x, df = deg_free),
      color = "lightblue",
      se = FALSE
    ) +
    labs(title = paste(deg_free, "Spline Terms"),
         y = "Sale Price (USD)")
}

(plot_smoother(2) + plot_smoother(5)) / (plot_smoother(20) + plot_smoother(100))


## Two terms underfit the data while 100 terms overfit.

## The panels with five and twenty terms seem like reasonably smooth fits that catch 
## the main patterns of the data. This indicates that the proper amount of 
## “nonlinear-ness” matters.

recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude,
       data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, deg_free = 20)

# (4.4) Feature Extraction

## Another common method for representing multiple features at once is called feature extraction. 
## Most of these techniques create new features from the predictors that capture the information 
## in the broader set as a whole. 

## For example, principal component analysis (PCA) tries to extract as much of the original 
## information in the predictor set as possible using a smaller number of features. 
## PCA is a linear extraction method, meaning that each new feature is a linear combination 
## of the original predictors. One nice aspect of PCA is that each of the new features, 
## called the principal components or PCA scores, are uncorrelated with one another.

## Because of this, PCA can be very effective at reducing the correlation between predictors. 
## Note that PCA is only aware of the predictors; the new PCA features might not be associated 
## with the outcome.

## In the Ames data, several predictors measure size of the property, such as the total 
## basement size (Total_Bsmt_SF), size of the first floor (First_Flr_SF), the gross living 
## area (Gr_Liv_Area), and so on. 

## PCA might be an option to represent these potentially 
## redundant variables as a smaller feature set. Apart from the gross living area, these 
## predictors have the suffix SF in their names (for square feet) so a recipe step for PCA 
## might look like:

# Use a regular expression to capture house size predictors: 
step_pca(dplyr::select(matches("(SF$)|(Gr_Liv)")))

## Note that all of these columns are measured in square feet. PCA assumes that all of the 
## predictors are on the same scale. That’s true in this case, but often this step can be 
## preceded by step_normalize(), which will center and scale each column.

## There are existing recipe steps for other extraction methods, such as: independent component 
## analysis (ICA), non-negative matrix factorization (NNMF), multidimensional scaling (MDS), 
## uniform manifold approximation and projection (UMAP), and others.

# (4.5) Row sampling steps

## Recipe steps can affect the rows of a data set as well. For example, subsampling techniques 
## for class imbalances change the class proportions in the data being given to the model; 
## these techniques often don’t improve overall performance but can generate better behaved 
## distributions of the predicted class probabilities. 

  ## Downsampling the data keeps the minority class and takes a random sample of the majority 
  ## class so that class frequencies are balanced.

  ## Upsampling replicates samples from the minority class to balance the classes. Some 
  ## techniques do this by synthesizing new samples that resemble the minority class data 
  ## while other methods simply add the same minority samples repeatedly.

  ## Hybrid methods do a combination of both.

## Only the training set should be affected by these techniques. The test set or other holdout 
## samples should be left as-is when processed using the recipe. For this reason, all of the 
## subsampling steps default the skip argument to have a value of TRUE.

## Other step functions are row-based as well: step_filter(), step_sample(), step_slice(), 
## and step_arrange(). In almost all uses of these steps, the skip argument should be set to TRUE

# (4.6) General transformations

## step_mutate() can be used to conduct a variety of basic operations to the data. 
## It is best used for straightforward transformations like computing a ratio of two variables, 
## such as Bedroom_AbvGr / Full_Bath, the ratio of bedrooms to bathrooms for the Ames housing data.

## When using this flexible step, use extra care to avoid data leakage in your preprocessing. 
## Consider, for example, the transformation x = w > mean(w). When applied to new data or 
## testing data, this transformation would use the mean of w from the new data, not the 
## mean of w from the training data.

# (4.7) NLP

## The textrecipes package can apply natural language processing methods to the data. 
## The input column is typically a string of text, and different steps can be used to tokenize 
## the data (e.g., split the text into separate words), filter out tokens, and create new features 
## appropriate for modeling.

# (5) Skipping steps for new data

## The sale price data are already log-transformed in the Ames data frame.

## This will cause a failure when the recipe is applied to new properties with an unknown 
## sale price. Since price is what we are trying to predict, there probably won’t be a column 
## in the data for this variable. In fact, to avoid information leakage, many tidymodels 
## packages isolate the data being used when making any predictions. This means that the 
## training set and any outcome columns are not available for use at prediction time.

## For simple transformations of the outcome column(s), we strongly suggest that 
## those operations be conducted outside of the recipe.

## At the time of this writing, the step functions in the recipes and themis packages that 
## are only applied to the training data are: step_adasyn(), step_bsmote(), step_downsample(), 
## step_filter(), step_naomit(), step_nearmiss(), step_rose(), step_sample(), step_slice(), 
## step_smote(), step_smotenc(), step_tomek(), and step_upsample().

# (6) Tidy a recipe()

## In Section 3.3, we introduced the tidy() verb for statistical objects. There is also a 
## tidy() method for recipes, as well as individual recipe steps.

ames_rec <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)

####
## The tidy() method, when called with the recipe object, gives a summary of the recipe steps:
####

tidy(ames_rec)

ames_rec <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01, id = "my_id") %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)

## We’ll refit the workflow with this new recipe:

lm_wflow <- 
  workflow() %>% 
  add_model(lm_model) %>% 
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)


## The tidy() method can be called again along with the id identifier we specified to get our 
## results for applying step_other():

estimated_recipe <- 
  lm_fit %>% 
  extract_recipe(estimated = TRUE)

tidy(estimated_recipe, id = "my_id")

tidy(estimated_recipe, number = 2)

## Each tidy() method returns the relevant information about that step. For example, the 
## tidy() method for step_dummy() returns a column with the variables that were converted to 
## dummy variables and another column with all of the known levels for each column.

# (7) Column roles

## When a formula is used with the initial call to recipe() it assigns roles to each of 
## the columns, depending on which side of the tilde they are on. Those roles are 
## either "predictor" or "outcome". However, other roles can be assigned as needed.

##  In other words, the column could be important even when it isn’t a predictor or outcome.

## To solve this, the add_role(), remove_role(), and update_role() functions can be helpful. 
## For example, for the house price data, the role of the street address column could be 
## modified using:

## ames_rec %>% update_role(address, new_role = "street address")

# (F) Judging Model Effectiveness

## Once we have a model, we need to know how well it works. A quantitative approach for 
## estimating effectiveness allows us to understand the model, to compare different models, 
## or to tweak the model to improve performance. Our focus in tidymodels is on empirical 
## validation; this usually means using data that were not used to create the model as the 
## substrate to measure effectiveness.

## The best approach to empirical validation involves using resampling methods

## When judging model effectiveness, your decision about which metrics to examine can be 
## critical. In later chapters, certain model parameters will be empirically optimized and a 
## primary performance metric will be used to choose the best sub-model. Choosing the wrong 
## metric can easily result in unintended consequences. For example, two common metrics for 
## regression models are the root mean squared error (RMSE) and the coefficient of determination
## (a.k.a. Rsquared). The former measures accuracy while the latter measures correlation.

# (9.1) Performance Metrics and Performance

## The effectiveness of any given model depends on how the model will be used. An inferential 
## model is used primarily to understand relationships, and typically emphasizes the 
## choice (and validity) of probabilistic distributions and other generative qualities that 
## define the model. For a model used primarily for prediction, by contrast, predictive strength
## is of primary importance and other concerns about underlying statistical qualities may be 
## less important. Predictive strength is usually determined by how close our predictions come 
## to the observed data, i.e., fidelity of the model predictions to the actual results. This 
## chapter focuses on functions that can be used to measure predictive strength.

# (9.2) Regression Metrics

## The functions in the yardstick package that produce performance metrics have 
## consistent interfaces.

## This model lm_wflow_fit combines a linear regression model with a predictor set supplemented
## with an interaction and spline functions for longitude and latitude.

ames_test_res <- predict(lm_fit, new_data = ames_test %>% select(-Sale_Price))
ames_test_res

## The predicted numeric outcome from the regression model is named .pred. 
## Let’s match the predicted values with their corresponding observed outcome values:

ames_test_res <- bind_cols(ames_test_res, ames_test %>% select(Sale_Price))
ames_test_res

## It is best practice to analyze the predictions on the transformed scale 
## (if one were used) even if the predictions are reported using the original units.

ggplot(ames_test_res, aes(x = Sale_Price, y = .pred)) + 
  # Create a diagonal line:
  geom_abline(lty = 2) + 
  geom_point(alpha = 0.5) + 
  labs(y = "Predicted Sale Price (log10)", x = "Sale Price (log10)") +
  # Scale and size the x- and y-axis uniformly:
  coord_obs_pred()

## Let’s compute the root mean squared error for this model using the rmse() function:

rmse(ames_test_res, truth = Sale_Price, estimate = .pred)

## To compute multiple metrics at once, we can create a metric set.
## Let’s add Rsquared and the mean absolute error:

ames_metrics <- metric_set(rmse, rsq, mae)
ames_metrics(ames_test_res, truth = Sale_Price, estimate = .pred)

# (9.3) Binary Classification Metrics

## To illustrate other ways to measure model performance, we will switch to a different example. 

data(two_class_example)
tibble(two_class_example)

# A confusion matrix: 
conf_mat(two_class_example, truth = truth, estimate = predicted)

# Accuracy:
accuracy(two_class_example, truth, predicted)

# Matthews correlation coefficient:
mcc(two_class_example, truth, predicted)

# F1 metric:
f_meas(two_class_example, truth, predicted)

# Combining these three classification metrics together
classification_metrics <- metric_set(accuracy, mcc, f_meas)
classification_metrics(two_class_example, truth = truth, estimate = predicted)

## The Matthews correlation coefficient and F1 score both summarize the confusion matrix, 
## but compared to mcc(), which measures the quality of both positive and negative examples, 
## the f_meas() metric emphasizes the positive class, i.e., the event of interest.

f_meas(two_class_example, truth, predicted, event_level = "second")

## There are numerous classification metrics that use the predicted probabilities as inputs 
## rather than the hard class predictions. For example, the receiver operating 
## characteristic (ROC) curve computes the sensitivity and specificity over a continuum of 
## different event thresholds. The predicted class column is not used. There are two yardstick 
## functions for this method: roc_curve() computes the data points that make up the ROC curve 
## and roc_auc() computes the area under the curve.

## For two-class problems, the probability column for the event of interest is 
## passed into the function:

two_class_curve <- roc_curve(two_class_example, truth, Class1)
two_class_curve

roc_auc(two_class_example, truth, Class1)

## There is an autoplot() method that will take care of the details:

autoplot(two_class_curve)

## There are a number of other functions that use probability estimates, 
## including gain_curve(), lift_curve(), and pr_curve().

# (9.4) Multiclass Classification Metrics

## What about data with three or more classes? To demonstrate, let’s 
## explore a different example data set that has four classes:

data(hpc_cv)
tibble(hpc_cv)

## The functions for metrics that use the discrete class predictions are identical 
## to their binary counterparts:

accuracy(hpc_cv, obs, pred)

mcc(hpc_cv, obs, pred)

## There are wrapper methods that can be used to apply sensitivity to our four-class outcome. 
## These options are macro-averaging, macro-weighted averaging, and micro-averaging:

## Macro-averaging computes a set of one-versus-all metrics using the standard two-class 
## statistics. These are averaged.

## Macro-weighted averaging does the same but the average is weighted by the number of 
## samples in each class.

## Micro-averaging computes the contribution for each class, aggregates them, then 
## computes a single metric from the aggregates.

## Using sensitivity as an example, the usual two-class calculation is the ratio of the 
## number of correctly predicted events divided by the number of true events. The manual 
## calculations for these averaging methods are:

class_totals <- 
  count(hpc_cv, obs, name = "totals") %>% 
  mutate(class_wts = totals / sum(totals))
class_totals

cell_counts <- 
  hpc_cv %>% 
  group_by(obs, pred) %>% 
  count() %>% 
  ungroup()

# Compute the four sensitivities using 1-vs-all
one_versus_all <- 
  cell_counts %>% 
  filter(obs == pred) %>% 
  full_join(class_totals, by = "obs") %>% 
  mutate(sens = n / totals)
one_versus_all

# Three different estimates:
one_versus_all %>% 
  summarize(
    macro = mean(sens), 
    macro_wts = weighted.mean(sens, class_wts),
    micro = sum(n) / sum(totals)
  )

## Thankfully, there is no need to manually implement these averaging methods. 
## Instead, yardstick functions can automatically apply these methods via 
## the estimator argument:

sensitivity(hpc_cv, obs, pred, estimator = "macro")
sensitivity(hpc_cv, obs, pred, estimator = "macro_weighted")
sensitivity(hpc_cv, obs, pred, estimator = "micro")

## When dealing with probability estimates, there are some metrics with multiclass analogs. 

roc_auc(hpc_cv, obs, VF, F, M, L)

## Macro-weighted averaging is also available as an option for applying this 
## metric to a multiclass outcome:

roc_auc(hpc_cv, obs, VF, F, M, L, estimator = "macro_weighted")

hpc_cv %>% 
  group_by(Resample) %>% 
  accuracy(obs, pred)

## The groupings also translate to the autoplot() methods

# Four 1-vs-all ROC curves for each fold
hpc_cv %>% 
  group_by(Resample) %>% 
  roc_curve(obs, VF, F, M, L) %>% 
  autoplot()


  
  