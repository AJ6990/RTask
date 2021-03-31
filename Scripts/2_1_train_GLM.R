# Run and optimise a GLM
# 2 cases should be separately run and saved differently

library(tidyverse)
library(MLmetrics)
library(caret)

project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'

# case1: categorical with 3 or 4 levels with weight of evidence encoding
# Loading and transforming the data
source(file.path(project_path, 'Scripts/1_2_1_preprocessingForCase1.R'))

# case2: categorical with 3 or 4 levels with dummy encoding
# Loading and transforming the data
source(file.path(project_path, 'Scripts/1_2_2_preprocessingForCase2.R'))

# Custom glm function that can optimize the weights
source(file.path(project_path, 'Functions/case_weight_optimizer.R'))

# Parameters for the model training
train_control <- trainControl(method = 'repeatedcv', 
                              number = 5,
                              repeats = 3,
                              summaryFunction = binarySummary, 
                              classProbs = TRUE,
                              returnResamp = 'final',
                              savePredictions = 'final',
                              verboseIter = T)


# Train a GLM model
# The weight parameter is the weight ratio (weight minority / weight majority).
# Cases weight are defined such as the sum of the weight = length weight (makes coeff comparable)
trained_list <- lapply(formula_list, function(formula){
  set.seed(999) # Should be the same for all the algorithms, so that the folds are all the same
  trained_model <- train(recipe(formula,
                                data = trainingData),
                         data = trainingData,
                         metric = 'AUC',
                         tuneGrid = expand.grid(weight = seq(15)),
                         trControl = train_control,
                         method = weighted_glm,
                         family = 'binomial')
})

# The trained model is saved
saveRDS(list(trained_list = trained_list,
             testData = testData,
             trainingData = trainingData), 
        file = file.path(project_path, 'Data/trained_glm_case2.rds'))
