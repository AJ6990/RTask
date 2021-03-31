# Run and optimise a GBM

library(caret)
library(tidyverse)
library(recipes)

memory.limit(56000)

project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'

# Loading and transforming the data
source(file.path(project_path, 'Scripts/1_2_2_preprocessingForCase2.R'))

# # Loading and transforming the data (with tree preprocessing)
# source(file.path(project_path, 'Scripts/1_1_preprocessingForTreeModels.R'))

# Parameters for the model training
train_control <- trainControl(method = 'repeatedcv', 
                              number = 5,
                              repeats = 3,
                              summaryFunction = prSummary, 
                              classProbs = TRUE,
                              savePredictions = "final",
                              verboseIter = F)



# Define the tuning parameter grid or hyperparameter grid
tgrid <- expand.grid(interaction.depth = c(3),
                     n.trees=seq(300, 500, by=100),
                     shrinkage = c(0.1, 0.2),
                     n.minobsinnode = c(2,3))

# Train a random forest model
trained_list <- lapply(formula_list, function(formula){
  set.seed(999) # Should be the same for all the algorithms, so that the folds are all the same
  trained_model <- train(recipe(formula,
                                data = trainingData),
                              data = trainingData,
                              metric = 'AUC',
                              tuneGrid = tgrid,
                              trControl = train_control,
                              method = 'gbm')

})
# The trained model is saved
saveRDS(list(trained_list = trained_list,
             testData = testData,
             trainingData = trainingData), 
        file = file.path(project_path, 'Data/trained_GBM.rds'))





