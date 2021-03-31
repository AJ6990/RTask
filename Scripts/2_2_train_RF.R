# Run and optimise a RF

library(caret)
library(tidyverse)
library(recipes)
library(ranger)

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
                              savePredictions = "final",
                              verboseIter = T)

# Define the tuning parameter grid or hyperparameter grid
tgrid <- expand.grid(
  mtry = seq(5),
  splitrule = "gini",
  min.node.size = 1)


# Train a random forest model
trained_list <- lapply(formula_list, function(formula){
  set.seed(999) # Should be the same for all the algorithms, so that the folds are all the same
  trained_model <- train(recipe(formula,
                                data = trainingData),
                              data = trainingData,
                              metric = 'AUC',
                              tuneGrid = tgrid,
                              trControl = train_control,
                              method = 'ranger',
                              num.threads = 7,
                              importance = 'impurity')
})

# The trained model is saved
saveRDS(list(trained_list = trained_list,
             testData = testData,
             trainingData = trainingData), 
        file = file.path(project_path, 'Data/trained_RF.rds'))


