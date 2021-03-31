# Run and optimise a XGB

library(tidyverse)
library(caret)
library(recipes)
library(xgboost)

memory.limit(56000)

project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'

# Loading and transforming the data
source(file.path(project_path, 'Scripts/1_2_2_preprocessingForCase2.R'))


# Parameters for the model training
train_control <- trainControl(method = 'repeatedcv', 
                              number = 5,
                              repeats = 3,
                              summaryFunction = prSummary, 
                              classProbs = TRUE,
                              savePredictions = 'final',
                              verboseIter = T)

# Define the tuning parameter grid or hyperparameter grid
tgrid <- expand.grid(eta = seq(0.05, 0.15, by = 0.05),
                     max_depth = c(5, 6),
                     min_child_weight = c(5, 7),
                     gamma = seq(0, 0.4, by = 0.1),
                     colsample_bytree = 0.5,
                     nrounds = 100,
                     subsample = 0.8)


trained_list <- lapply(formula_list, function(formula){
  # Prepare data for XGB classifier
  recipeObject <- recipe(formula,
                    data = trainingData) 
  
  prepareRecipe <- prep(recipeObject, training = trainingData)
  bakedTrain <- bake(prepareRecipe, trainingData)
  
  set.seed(999) # Should be the same for all the algorithms, so that the folds are all the same
  trained_model <- train(label ~ .,
                       data = bakedTrain,
                       metric = 'AUC',
                       tuneGrid = tgrid,
                       trControl = train_control,
                       method = "xgbTree")

})

# The trained model is saved
saveRDS(list(trained_list = trained_list,
             testData = testData,
             trainingData = trainingData), 
        file = file.path(project_path, 'Data/trained_xgb.rds'))
