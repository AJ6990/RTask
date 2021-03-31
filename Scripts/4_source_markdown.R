## Creates all the objects necessary for the report
## Has to be created there because of memory issues

library(caret)
library(tidyverse)
library(yarrr)
library(formula.tools)
library(kableExtra)

memory.limit(56000)

project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'


source('C:/Users/Alizadeh/Documents/Altran/Recruiting_Task/Functions/model_metricsglobal.R')

# trained models
##  some of them are not continued furthermore because of performance
# glm1_list <- readRDS(file.path(project_path, 'Data/trained_glm_case1.rds'))
glm2_list <- readRDS(file.path(project_path, 'Data/trained_glm_case2.rds'))
rf_list <- readRDS(file.path(project_path, 'Data/trained_RF.rds'))
gbm_list <- readRDS(file.path(project_path, 'Data/trained_GBM.rds'))
# svm1_list <- readRDS(file.path(project_path, 'Data/trained_SVMcase1.rds'))
svm2_list <- readRDS(file.path(project_path, 'Data/trained_SVMcase2.rds'))
# xgb_list <- readRDS(file.path(project_path, 'Data/trained_xgb.rds'))
dt_list <- readRDS(file.path(project_path, 'Data/trained_DT.rds'))

# Previously used test and train as its same for all algorithms
testSet <- gbm_list$testData
trainingSet <- gbm_list$trainingData

# Model training
## Cross-validation metrics
## cv AUC and prAUC (values and plots)
## Save a list with the optimal model base on ROC AUC

models <- list('GLM' = glm2_list,
               'RF' = rf_list,
               'GBM' = gbm_list,
               'SVM' = svm2_list,
               'DT' = dt_list
)

rm(list = c('glm2_list', 'rf_list', 'gbm_list','svm2_list','dt_list'))
gc()

# CV metrics: runs the metrics for the CV predictions
# cv AUC and prAUC (values and plots)
# Save a list with the optimal model based on ROC AUC
crossValidationList <- sapply(names(models), function(name_algo){
  model_metrics(trained_list = models[[name_algo]]$trained_list,
                calibrating = F,
                y_name = 'label',
                discrimination_metrics = c('AUC', 'prAUC'))
},
USE.NAMES = T,
simplify = F)


# Determining optimal model and saves the associated optimal threshold
optimal_models <- lapply(names(crossValidationList), function(name_algo){
  metrics <- crossValidationList[[name_algo]]                 
  
  # Extract the AUC and prAUC values to print them as part of the legend
  aucs <- metrics$discrimination_metric %>% 
    bind_rows(sapply(metrics$predictions, function(x) unique(x$threshold))) %>% 
    mutate(Metric = replace_na(as.character(Metric), 'threshold')) %>% 
    pivot_longer(cols = -Metric) %>% 
    pivot_wider(names_from = Metric, values_from = value) %>% 
    mutate(algo = name_algo) %>% 
    filter(AUC == max(AUC)) %>%
    dplyr::slice(1L)
}) %>% 
  bind_rows()


## Keep only one model per algorithm, based on AUC
list_best_models <- sapply(optimal_models$algo, function(name_algo){
  name_model <- optimal_models %>% 
    filter(algo == name_algo) %>% 
    pull(name)
  
  models[[name_algo]]$trained_list[[name_model]]
},
USE.NAMES = T,
simplify = F)

rm(models)
gc()

## Model validation
thresholds <- optimal_models$threshold
names(thresholds) <- optimal_models$algo

prauc <- optimal_models$prAUC
weights <- prauc / sum(prauc)

list_metrics<- model_metrics(trained_list = list_best_models,
                              data = testSet,
                              trainingData = trainingSet,
                              y_name = 'label',
                              threshold = thresholds,
                              discrimination_metrics = c('AUC','F1', 'Recall',
                                                         'Precision'))
