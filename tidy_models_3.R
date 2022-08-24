# Experimenting with machine learning in R with tidymodels and the Kaggle titanic dataset
# https://oliviergimenez.github.io/learning-machine-learning/

## (1 and 2) Load the required packages

library(tidymodels) # metapackage for ML 
library(tidyverse) # metapackage for data manipulation and visualisation
# install.packages("stacks")
library(stacks) # stack ML models for better perfomance
theme_set(theme_light())
doParallel::registerDoParallel(cores = 4) # parallel computations
# install.packages("naniar")
library(naniar)


## (3) Load the data 

## Read in the training data

rawdata <- read_csv("C:/R_Files/tidymodels/titanic/train.csv")
glimpse(rawdata)

naniar::miss_var_summary(rawdata)

# Clean the data

# Get most frequent port of embarkation
uniqx <- unique(na.omit(rawdata$Embarked))
mode_embarked <- as.character(fct_drop(uniqx[which.max(tabulate(match(rawdata$Embarked, uniqx)))]))


# Build function for data cleaning and handling NAs
process_data <- function(tbl){
  
  tbl %>%
    mutate(class = case_when(Pclass == 1 ~ "first",
                             Pclass == 2 ~ "second",
                             Pclass == 3 ~ "third"),
           class = as_factor(class),
           gender = factor(Sex),
           fare = Fare,
           age = Age,
           ticket = Ticket,
           alone = if_else(SibSp + Parch == 0, "yes", "no"), # alone variable
           alone = as_factor(alone),
           port = factor(Embarked), # rename embarked as port
           title = str_extract(Name, "[A-Za-z]+\\."), # title variable
           title = fct_lump(title, 4)) %>% # keep only most frequent levels of title
    mutate(port = ifelse(is.na(port), mode_embarked, port), # deal w/ NAs in port (replace by mode)
           port = as_factor(port)) %>%
    group_by(title) %>%
    mutate(median_age_title = median(age, na.rm = T)) %>%
    ungroup() %>%
    mutate(age = if_else(is.na(age), median_age_title, age)) %>% # deal w/ NAs in age (replace by median in title)
    mutate(ticketfreq = ave(1:nrow(.), FUN = length),
           fareadjusted = fare / ticketfreq) %>%
    mutate(familyage = SibSp + Parch + 1 + age/70)
  
}

# Process the data
dataset <- rawdata %>%
  process_data() %>%
  mutate(survived = as_factor(if_else(Survived == 1, "yes", "no"))) %>%
  mutate(survived = relevel(survived, ref = "yes")) %>% # first event is survived = yes
  select(survived, class, gender, age, alone, port, title, fareadjusted, familyage) 

# Have a look again
glimpse(dataset)

naniar::miss_var_summary(dataset)


# Let’s apply the same treatment to the test dataset.

rawdata <- read_csv("C:/R_Files/tidymodels/titanic/test.csv") 
holdout <- rawdata %>%
  process_data() %>%
  select(PassengerId, class, gender, age, alone, port, title, fareadjusted, familyage) 

glimpse(holdout)

naniar::miss_var_summary(holdout)

## (4) Exploratory data analysis

skimr::skim(dataset)

dataset %>%
  count(survived)

dataset %>%
  group_by(gender) %>%
  summarize(n = n(),
            n_surv = sum(survived == "yes"),
            pct_surv = n_surv / n)

dataset %>%
  group_by(title) %>%
  summarize(n = n(),
            n_surv = sum(survived == "yes"),
            pct_surv = n_surv / n) %>%
  arrange(desc(pct_surv))

dataset %>%
  group_by(class, gender) %>%
  summarize(n = n(),
            n_surv = sum(survived == "yes"),
            pct_surv = n_surv / n) %>%
  arrange(desc(pct_surv))

# Informative graphs

dataset %>%
  group_by(class, gender) %>%
  summarize(n = n(),
            n_surv = sum(survived == "yes"),
            pct_surv = n_surv / n) %>%
  mutate(class = fct_reorder(class, pct_surv)) %>%
  ggplot(aes(pct_surv, class, fill = class, color = class)) +
  geom_col(position = position_dodge()) +
  scale_x_continuous(labels = percent) +
  labs(x = "% in category that survived", fill = NULL, color = NULL, y = NULL) +
  facet_wrap(~gender)

dataset %>%
  mutate(age = cut(age, breaks = c(0, 20, 40, 60, 80))) %>%
  group_by(age, gender) %>%
  summarize(n = n(),
            n_surv = sum(survived == "yes"),
            pct_surv = n_surv / n) %>%
  mutate(age = fct_reorder(age, pct_surv)) %>%
  ggplot(aes(pct_surv, age, fill = age, color = age)) +
  geom_col(position = position_dodge()) +
  scale_x_continuous(labels = percent) +
  labs(x = "% in category that survived", fill = NULL, color = NULL, y = NULL) +
  facet_wrap(~gender)

dataset %>%
  ggplot(aes(fareadjusted, group = survived, color = survived, fill = survived)) +
  geom_histogram(alpha = .4, position = position_dodge()) +
  labs(x = "fare", y = NULL, color = "survived?", fill = "survived?")

dataset %>%
  ggplot(aes(familyage, group = survived, color = survived, fill = survived)) +
  geom_histogram(alpha = .4, position = position_dodge()) +
  labs(x = "family aged", y = NULL, color = "survived?", fill = "survived?")

## (5) Training/testing datasets

# Split our dataset in two, one dataset for training and the other one for testing. 
# We will use an additionnal splitting step for cross-validation.

set.seed(2021)
spl <- initial_split(dataset, strata = "survived")
train <- training(spl)
test <- testing(spl)

train_5fold <- train %>%
  vfold_cv(5)

## (6) Gradient boosting algorithms - xgboost

## (6.1) Tuning

## Set up defaults.

mset <- metric_set(accuracy) # metric is accuracy
control <- control_grid(save_workflow = TRUE,
                        save_pred = TRUE,
                        extract = extract_model) # grid for tuning

## First a recipe.

xg_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)

## Then specify a gradient boosting model.

xg_model <- boost_tree(mode = "classification", # binary response
                       trees = tune(),
                       mtry = tune(),
                       tree_depth = tune(),
                       learn_rate = tune(),
                       loss_reduction = tune(),
                       min_n = tune()) # parameters to be tuned

## Now set our workflow.

xg_wf <- 
  workflow() %>% 
  add_model(xg_model) %>% 
  add_recipe(xg_rec)

## Use cross-validation to evaluate our model with different param config.

xg_tune <- xg_wf %>%
  tune_grid(train_5fold,
            metrics = mset,
            control = control,
            grid = crossing(trees = 1000,
                            mtry = c(3, 5, 8), # finalize(mtry(), train)
                            tree_depth = c(5, 10, 15),
                            learn_rate = c(0.01, 0.005),
                            loss_reduction = c(0.01, 0.1, 1),
                            min_n = c(2, 10, 25)))

## Visualize the results.

autoplot(xg_tune) + theme_light()

## Collect metrics.

xg_tune %>%
  collect_metrics() %>%
  arrange(desc(mean))

## The tuning takes some time. There are other ways to explore the parameter space more 
## efficiently. For example, we will use the function dials::grid_max_entropy() in the 
## last section about ensemble modelling. Here, I will use finetune::tune_race_anova.

install.packages("finetune")

library(finetune)

xg_tune <-
  xg_wf %>%
  tune_race_anova(
    train_5fold,
    grid = 50,
    param_info = xg_model %>% parameters(),
    metrics = metric_set(accuracy),
    control = control_race(verbose_elim = TRUE))

## Visualize the results.


autoplot(xg_tune)


## Collect metrics.

xg_tune %>%
  collect_metrics() %>%
  arrange(desc(mean))


## (6.2) Fit model


xg_fit <- xg_wf %>%
  finalize_workflow(select_best(xg_tune)) %>%
  fit(train)


## Check out accuracy on testing dataset to see if we overfitted.

xg_fit %>%
  augment(test, type.predict = "response") %>%
  accuracy(survived, .pred_class)

## Check out important features (aka predictors).

importances <- xgboost::xgb.importance(model = extract_fit_engine(xg_fit))
importances %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(Gain, Feature)) +
  geom_col()

## (6.3) Make predictions

## Now we’re ready to predict survival for the holdout dataset and submit to Kaggle. 
## Note that I use the whole dataset, not just the training dataset.

xg_wf %>%
  finalize_workflow(select_best(xg_tune)) %>%
  fit(dataset) %>%
  augment(holdout) %>%
  select(PassengerId, Survived = .pred_class) %>%
  mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
  write_csv("titanic/output/xgboost.csv")

## Accuracy = ....


## (7) Random forests

## (7.1) Tuning

## First a recipe.

rf_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)

## Then specify a random forest model.

rf_model <- rand_forest(mode = "classification", # binary response
                        engine = "ranger", # by default
                        mtry = tune(),
                        trees = tune(),
                        min_n = tune()) # parameters to be tuned

## Now set our workflow.

rf_wf <- 
  workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(rf_rec)

## Use cross-validation to evaluate our model with different param config.

rf_tune <-
  rf_wf %>%
  tune_race_anova(
    train_5fold,
    grid = 50,
    param_info = rf_model %>% parameters(),
    metrics = metric_set(accuracy),
    control = control_race(verbose_elim = TRUE))

## Visualize the results.

autoplot(rf_tune)

## Collect metrics.

rf_tune %>%
  collect_metrics() %>%
  arrange(desc(mean))

## (7.2) Fit model

## Use best config to fit model to training data.

rf_fit <- rf_wf %>%
  finalize_workflow(select_best(rf_tune)) %>%
  fit(train)

## Check out accuracy on testing dataset to see if we overfitted.

rf_fit %>%
  augment(test, type.predict = "response") %>%
  accuracy(survived, .pred_class)

## Check out important features (aka predictors).
# install.packages("vip")
library(vip)
finalize_model(
  x = rf_model,
  parameters = select_best(rf_tune)) %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(survived ~ ., data = juice(prep(rf_rec))) %>%
  vip(geom = "point")

## 7.3 Make predictions

## Now we’re ready to predict survival for the holdout dataset and submit to Kaggle.

rf_wf %>%
  finalize_workflow(select_best(rf_tune)) %>%
  fit(dataset) %>%
  augment(holdout) %>%
  select(PassengerId, Survived = .pred_class) %>%
  mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
  write_csv("titanic/output/randomforest.csv")


## Accuracy = ....


## (8) Gradient boosting algorithms - catboost !!!!SWITCHED TO lightgbm!!!!

## (8.1) Tuning

## Set up defaults.

mset <- metric_set(accuracy) # metric is accuracy
control <- control_grid(save_workflow = TRUE,
                        save_pred = TRUE,
                        extract = extract_model) # grid for tuning

## First a recipe.

cb_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)

## Then specify a cat boosting model.

# remotes::install_github("curso-r/treesnip")

library(treesnip)
cb_model <- boost_tree(mode = "classification",
                       engine = "spark",
                       mtry = tune(),
                       trees = tune(),
                       min_n = tune(),
                       tree_depth = tune(),
                       learn_rate = tune()) # parameters to be tuned

## Now set our workflow.

cb_wf <- 
  workflow() %>% 
  add_model(cb_model) %>% 
  add_recipe(cb_rec)

## Use cross-validation to evaluate our model with different param config.

### Not working!!!

cb_tune <- cb_wf %>%
  tune_race_anova(
    train_5fold,
    grid = 30,
    param_info = cb_model %>% parameters(),
    metrics = metric_set(accuracy),
    control = control_race(verbose_elim = TRUE))

## Visualize the results.

autoplot(cb_tune)

## Collect metrics.

cb_tune %>%
  collect_metrics() %>%
  arrange(desc(mean))

## (8.2) Fit model

## Use best config to fit model to training data.

cb_fit <- cb_wf %>%
  finalize_workflow(select_best(cb_tune)) %>%
  fit(train)

## Check out accuracy on testing dataset to see if we overfitted.

cb_fit %>%
  augment(test, type.predict = "response") %>%
  accuracy(survived, .pred_class)

## (8.3) Make predictions

## Now we’re ready to predict survival for the holdout dataset and submit to Kaggle.

cb_wf %>%
  finalize_workflow(select_best(cb_tune)) %>%
  fit(dataset) %>%
  augment(holdout) %>%
  select(PassengerId, Survived = .pred_class) %>%
  mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
  write_csv("titanic/output/catboost.csv")

## (9) Regularization methods

## Let’s continue with elastic net regularization.

## (9.1) Tuning

## First a recipe.

en_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_normalize(all_numeric_predictors()) %>% # normalize
  step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)

## Then specify a regularization model. We tune parameter mixture, with 
## ridge regression for mixture = 0, and lasso for mixture = 1.

en_model <- logistic_reg(penalty = tune(), 
                         mixture = tune()) %>% # param to be tuned
  set_engine("glmnet") %>% # elastic net
  set_mode("classification") # binary response

## Now set our workflow.

en_wf <- 
  workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(en_rec)

## Use cross-validation to evaluate our model with different param config.

en_tune <- en_wf %>%
  tune_grid(train_5fold,
            metrics = mset,
            control = control,
            grid = crossing(penalty = 10 ^ seq(-8, -.5, .5),
                            mixture = seq(0, 1, length.out = 10)))

## Visualize the results.

autoplot(en_tune)

## Collect metrics.

en_tune %>%
  collect_metrics() %>%
  arrange(desc(mean))


## (9.2) Fit model
## Use best config to fit model to training data.

en_fit <- en_wf %>%
  finalize_workflow(select_best(en_tune)) %>%
  fit(train)

## Check out accuracy on testing dataset to see if we overfitted.

en_fit %>%
  augment(test, type.predict = "response") %>%
  accuracy(survived, .pred_class)

## Check out important features (aka predictors).

library(broom)
en_fit$fit$fit$fit %>%
  tidy() %>%
  filter(lambda >= select_best(en_tune)$penalty) %>%
  filter(lambda == min(lambda),
         term != "(Intercept)") %>%
  mutate(term = fct_reorder(term, estimate)) %>%
  ggplot(aes(estimate, term, fill = estimate > 0)) +
  geom_col() +
  theme(legend.position = "none")


## 9.3 Make predictions
## Now we’re ready to predict survival for the holdout dataset and submit to Kaggle.

en_wf %>%
  finalize_workflow(select_best(en_tune)) %>%
  fit(dataset) %>%
  augment(holdout) %>%
  select(PassengerId, Survived = .pred_class) %>%
  mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
  write_csv("titanic/output/regularization.csv")

## Accuracy is ...


## (10) Logistic regression

## And what about a good old-fashioned logistic regression (not a ML algo)?
  
## First a recipe.

logistic_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_normalize(all_numeric_predictors()) %>% # normalize
  step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)

## Then specify a logistic regression.

logistic_model <- logistic_reg() %>% # no param to be tuned
  set_engine("glm") %>% # elastic net
  set_mode("classification") # binary response

## Now set our workflow.

logistic_wf <- 
  workflow() %>% 
  add_model(logistic_model) %>% 
  add_recipe(logistic_rec)

## Fit model.

logistic_fit <- logistic_wf %>%
  fit(train)

## Inspect significant features (aka predictors).

tidy(logistic_fit, exponentiate = TRUE) %>%
  filter(p.value < 0.05)

## Same thing, but graphically.

library(broom)
logistic_fit %>%
  tidy() %>%
  mutate(term = fct_reorder(term, estimate)) %>%
  ggplot(aes(estimate, term, fill = estimate > 0)) +
  geom_col() +
  theme(legend.position = "none")

## Check out accuracy on testing dataset to see if we overfitted.

logistic_fit %>%
  augment(test, type.predict = "response") %>%
  accuracy(survived, .pred_class)


# Confusion matrix.

logistic_fit %>%
  augment(test, type.predict = "response") %>%
  conf_mat(survived, .pred_class)


## ROC curve.

logistic_fit %>%
  augment(test, type.predict = "response") %>%
  roc_curve(truth = survived, estimate = .pred_yes) %>%
  autoplot()

##Now we’re ready to predict survival for the holdout dataset and submit to Kaggle.

logistic_wf %>%
  fit(dataset) %>%
  augment(holdout) %>%
  select(PassengerId, Survived = .pred_class) %>%
  mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
  write_csv("titanic/output/logistic.csv")


## (11) Neural networks
## We go on with neural networks.

## (11.1) Tuning

## Set up defaults.

mset <- metric_set(accuracy) # metric is accuracy
control <- control_grid(save_workflow = TRUE,
                        save_pred = TRUE,
                        extract = extract_model) # grid for tuning

## First a recipe.

nn_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)

## Then specify a neural network.

nn_model <- mlp(epochs = tune(), 
                hidden_units = tune(), 
                dropout = tune()) %>% # param to be tuned
  set_mode("classification") %>% # binary response var
  set_engine("keras", verbose = 0)

## Now set our workflow.

nn_wf <- 
  workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(nn_rec)

## Use cross-validation to evaluate our model with different param config.

## install.packages("keras")
library(keras)

nn_tune <- nn_wf %>%
  tune_race_anova(
    train_5fold,
    grid = 30,
    param_info = nn_model %>% parameters(),
    metrics = metric_set(accuracy),
    control = control_race(verbose_elim = TRUE))

## Visualize the results.

autoplot(nn_tune)

## Collect metrics.

nn_tune %>%
  collect_metrics() %>%
  arrange(desc(mean))


## (11.2) Fit model
## Use best config to fit model to training data.

nn_fit <- nn_wf %>%
  finalize_workflow(select_best(nn_tune)) %>%
  fit(train)

## Check out accuracy on testing dataset to see if we overfitted.

nn_fit %>%
  augment(test, type.predict = "response") %>%
  accuracy(survived, .pred_class)


## (11.3) Make predictions
## Now we’re ready to predict survival for the holdout dataset and submit to Kaggle.

nn_wf %>%
  finalize_workflow(select_best(nn_tune)) %>%
  fit(train) %>%
  augment(holdout) %>%
  select(PassengerId, Survived = .pred_class) %>%
  mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
  write_csv("output/titanic/nn.csv")


## (12) Support vector machines
## We go on with support vector machines.

## (12.1) Tuning
## Set up defaults.

mset <- metric_set(accuracy) # metric is accuracy
control <- control_grid(save_workflow = TRUE,
                        save_pred = TRUE,
                        extract = extract_model) # grid for tuning

## First a recipe.

svm_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  # remove any zero variance predictors
  step_zv(all_predictors()) %>% 
  # remove any linear combinations
  step_lincomb(all_numeric()) %>%
  step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)

## Then specify a svm.

svm_model <- svm_rbf(cost = tune(), 
                     rbf_sigma = tune()) %>% # param to be tuned
  set_mode("classification") %>% # binary response var
  set_engine("kernlab")

## Now set our workflow.

svm_wf <- 
  workflow() %>% 
  add_model(svm_model) %>% 
  add_recipe(svm_rec)

## Use cross-validation to evaluate our model with different param config.

svm_tune <- svm_wf %>%
  tune_race_anova(
    train_5fold,
    grid = 30,
    param_info = svm_model %>% parameters(),
    metrics = metric_set(accuracy),
    control = control_race(verbose_elim = TRUE))

## Visualize the results.

autoplot(svm_tune)


## Collect metrics.

svm_tune %>%
  collect_metrics() %>%
  arrange(desc(mean))


## (12.2) Fit model
## Use best config to fit model to training data.

svm_fit <- svm_wf %>%
  finalize_workflow(select_best(svm_tune)) %>%
  fit(train)

## Check out accuracy on testing dataset to see if we overfitted.

svm_fit %>%
  augment(test, type.predict = "response") %>%
  accuracy(survived, .pred_class)


## (12.3) Make predictions
## Now we’re ready to predict survival for the holdout dataset and submit to Kaggle.

svm_wf %>%
  finalize_workflow(select_best(svm_tune)) %>%
  fit(dataset) %>%
  augment(holdout) %>%
  select(PassengerId, Survived = .pred_class) %>%
  mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
  write_csv("titanic/output/svm.csv")


## (13) Decision trees
## We go on with decision trees.

## (13.1) Tuning
## Set up defaults.

mset <- metric_set(accuracy) # metric is accuracy
control <- control_grid(save_workflow = TRUE,
                        save_pred = TRUE,
                        extract = extract_model) # grid for tuning
## First a recipe.

dt_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_zv(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)

## Then specify a decision tree model.

# install.packages("baguette")
library(baguette)

dt_model <- bag_tree(cost_complexity = tune(),
                     tree_depth = tune(),
                     min_n = tune()) %>% # param to be tuned
  set_engine("rpart", times = 25) %>% # nb bootstraps
  set_mode("classification") # binary response var

## Now set our workflow.

dt_wf <- 
  workflow() %>% 
  add_model(dt_model) %>% 
  add_recipe(dt_rec)

## Use cross-validation to evaluate our model with different param config.

dt_tune <- dt_wf %>%
  tune_race_anova(
    train_5fold,
    grid = 30,
    param_info = dt_model %>% parameters(),
    metrics = metric_set(accuracy),
    control = control_race(verbose_elim = TRUE))

## Visualize the results.

autoplot(dt_tune)


## Collect metrics.

dt_tune %>%
  collect_metrics() %>%
  arrange(desc(mean))


## (13.2) Fit model
## Use best config to fit model to training data.

dt_fit <- dt_wf %>%
  finalize_workflow(select_best(dt_tune)) %>%
  fit(train)

## Check out accuracy on testing dataset to see if we overfitted.

dt_fit %>%
  augment(test, type.predict = "response") %>%
  accuracy(survived, .pred_class)


## (13.3) Make predictions
## Now we’re ready to predict survival for the holdout dataset and submit to Kaggle.

dt_wf %>%
  finalize_workflow(select_best(dt_tune)) %>%
  fit(dataset) %>%
  augment(holdout) %>%
  select(PassengerId, Survived = .pred_class) %>%
  mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
  write_csv("titanic/output/dt.csv")

#####################################################################################

## (14) Stacked ensemble modelling
## Let’s do some ensemble modelling with all algo but logistic and catboost. 
## Tune again with a probability-based metric. Start with xgboost.

library(finetune)
library(stacks)
# xgboost
xg_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_dummy(all_nominal_predictors()) # all factors var are split into binary terms (factor disj coding)
xg_model <- boost_tree(mode = "classification", # binary response
                       trees = tune(),
                       mtry = tune(),
                       tree_depth = tune(),
                       learn_rate = tune(),
                       loss_reduction = tune(),
                       min_n = tune()) # parameters to be tuned
xg_wf <- 
  workflow() %>% 
  add_model(xg_model) %>% 
  add_recipe(xg_rec)
xg_grid <- grid_latin_hypercube(
  trees(),
  finalize(mtry(), train),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  min_n(),
  size = 30)
xg_tune <- xg_wf %>%
  tune_grid(
    resamples = train_5fold,
    grid = xg_grid,
    metrics = metric_set(roc_auc),
    control = control_stack_grid())

## Then random forests.

# random forest
rf_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% 
  step_dummy(all_nominal_predictors()) 
rf_model <- rand_forest(mode = "classification", # binary response
                        engine = "ranger", # by default
                        mtry = tune(),
                        trees = tune(),
                        min_n = tune()) # parameters to be tuned
rf_wf <- 
  workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(rf_rec)
rf_grid <- grid_latin_hypercube(
  finalize(mtry(), train),
  trees(),
  min_n(),
  size = 30)
rf_tune <- rf_wf %>%
  tune_grid(
    resamples = train_5fold,
    grid = rf_grid,
    metrics = metric_set(roc_auc),
    control = control_stack_grid())

## Regularisation methods (between ridge and lasso).

# regularization methods
en_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_normalize(all_numeric_predictors()) %>% # normalize
  step_dummy(all_nominal_predictors()) 
en_model <- logistic_reg(penalty = tune(), 
                         mixture = tune()) %>% # param to be tuned
  set_engine("glmnet") %>% # elastic net
  set_mode("classification") # binary response
en_wf <- 
  workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(en_rec)
en_grid <- grid_latin_hypercube(
  penalty(),
  mixture(),
  size = 30)
en_tune <- en_wf %>%
  tune_grid(
    resamples = train_5fold,
    grid = en_grid,
    metrics = metric_set(roc_auc),
    control = control_stack_grid())

## Neural networks (takes time, so pick only a few values for illustration purpose).


# neural networks
nn_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% # replace missing value by median
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) 
nn_model <- mlp(epochs = tune(), 
                hidden_units = 2, 
                dropout = tune()) %>% # param to be tuned
  set_mode("classification") %>% # binary response var
  set_engine("keras", verbose = 0)
nn_wf <- 
  workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(nn_rec)
# nn_grid <- grid_latin_hypercube(
#   epochs(),
#   hidden_units(),
#   dropout(),
#   size = 10)
nn_tune <- nn_wf %>%
  tune_grid(
    resamples = train_5fold,
    grid = crossing(dropout = c(0.1, 0.2), epochs = c(250, 500, 1000)), # nn_grid
    metrics = metric_set(roc_auc),
    control = control_stack_grid())
autoplot(nn_tune)

# Support vector machines.

# support vector machines
svm_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% 
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) 
svm_model <- svm_rbf(cost = tune(), 
                     rbf_sigma = tune()) %>% # param to be tuned
  set_mode("classification") %>% # binary response var
  set_engine("kernlab")
svm_wf <- 
  workflow() %>% 
  add_model(svm_model) %>% 
  add_recipe(svm_rec)
svm_grid <- grid_latin_hypercube(
  cost(),
  rbf_sigma(),
  size = 30)
svm_tune <- svm_wf %>%
  tune_grid(
    resamples = train_5fold,
    grid = svm_grid,
    metrics = metric_set(roc_auc),
    control = control_stack_grid())

# Last, decision trees.

# decision trees
dt_rec <- recipe(survived ~ ., data = train) %>%
  step_impute_median(all_numeric()) %>% 
  step_zv(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 
library(baguette)
dt_model <- bag_tree(cost_complexity = tune(),
                     tree_depth = tune(),
                     min_n = tune()) %>% # param to be tuned
  set_engine("rpart", times = 25) %>% # nb bootstraps
  set_mode("classification") # binary response var
dt_wf <- 
  workflow() %>% 
  add_model(dt_model) %>% 
  add_recipe(dt_rec)
dt_grid <- grid_latin_hypercube(
  cost_complexity(),
  tree_depth(),
  min_n(),
  size = 30)
dt_tune <- dt_wf %>%
  tune_grid(
    resamples = train_5fold,
    grid = dt_grid,
    metrics = metric_set(roc_auc),
    control = control_stack_grid())
# Get best config.

xg_best <- xg_tune %>% filter_parameters(parameters = select_best(xg_tune))
rf_best <- rf_tune %>% filter_parameters(parameters = select_best(rf_tune))
en_best <- en_tune %>% filter_parameters(parameters = select_best(en_tune))
nn_best <- nn_tune %>% filter_parameters(parameters = select_best(nn_tune))
svm_best <- svm_tune %>% filter_parameters(parameters = select_best(svm_tune))
dt_best <- dt_tune %>% filter_parameters(parameters = select_best(dt_tune))


## Do the stacked ensemble modelling.

## Pile all models together.

blended <- stacks() %>% # initialize
  add_candidates(en_best) %>% # add regularization model
  add_candidates(xg_best) %>% # add gradient boosting
  add_candidates(rf_best) %>% # add random forest
  add_candidates(nn_best) %>% # add neural network
  add_candidates(svm_best) %>% # add svm
  add_candidates(dt_best) # add decision trees
blended


# Fit regularized model.

blended_fit <- blended %>%
  blend_predictions() # fit regularized model
# Visualise penalized model. Note that neural networks are dropped, 
# despite achieving best score when used in isolation. I’ll have to dig into that.

autoplot(blended_fit)


autoplot(blended_fit, type = "members")


autoplot(blended_fit, type = "weights")


# Fit candidate members with non-zero stacking coef with full training dataset.

blended_regularized <- blended_fit %>%
  fit_members() 
blended_regularized


# Perf on testing dataset?
  
  test %>%
  bind_cols(predict(blended_regularized, .)) %>%
  accuracy(survived, .pred_class)

# Now predict.

holdout %>%
  bind_cols(predict(blended_regularized, .)) %>%
  select(PassengerId, Survived = .pred_class) %>%
  mutate(Survived = if_else(Survived == "yes", 1, 0)) %>%
  write_csv("titanic/output/stacked.csv")
