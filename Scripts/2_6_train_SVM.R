# Run and optimise an SVM
# 2 cases should be separately run and saved differently

library(tidyverse)
library(MLmetrics)
library(recipes)
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


# Define the tuning parameter grid or hyperparameter grid
tgrid <- expand.grid(C = 2^(-1:0),
                     sigma = 2^(-2:-1))

# Train an SVM model
trained_list <- lapply(formula_list, function(formula){
  set.seed(999) # Should be the same for all the algorithms, so that the folds are all the same
  trained_model <- train(recipe(formula,
                                data = trainingData),
                       data = trainingData,
                       metric = 'AUC',
                       tuneGrid = tgrid,
                       trControl = train_control,
                       num.threads = 7,
                       method = "svmRadial")
})


# The trained model is saved
saveRDS(list(trained_list = trained_list,
             testData = testData,
             trainingData = trainingData), 
        file = file.path(project_path, 'Data/trained_SVMcase2.rds'))
