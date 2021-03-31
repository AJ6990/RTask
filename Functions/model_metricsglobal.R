# library(tidyverse)
library(dplyr)
library(caret)      
library(pROC)
library(formula.tools)
library(ROCR)
library(ranger)
library(xgboost)
library(DescTools)
library(matrixStats)
library(ResourceSelection)
library(Hmisc)
library(gbm)
library(boot)

########### Subfunctions ##############
# roc.df
## Creates a data frame with TPR (sensitivity) and FPR (1- specificity).
## Computes the Youden's J statistics (TPR - FPR) and uses it to define the ptimal threshold (maximum Youden's J)
## Computes AUC and AUC_CI (saved in one column each)
# dat: dataframe containing predicted probabilities and observed event in two seperate columns
# pred_name: name of the column containing the predicter probabilities
# obs_name: name of the column containing the observed event (factor, with event as the first level)
roc.df <- function(dat,
                   pred_name,
                   obs_name){
  
  obs <- dat[[obs_name]]
  pred <- dat[[pred_name]]
  
  roc <- roc(predictor = pred, response = obs, quiet = T)
  
  tibble(Threshold = roc$thresholds,
         Sensitivity = roc$sensitivities,
         `1 - Specificity` = 1 - roc$specificities,
         J = Sensitivity - `1 - Specificity`,
         optimal_threshold = J == max(J, na.rm = T),
         AUC = ci.auc(roc)[2],
         AUCLower = ci.auc(roc)[1],
         AUCUpper = ci.auc(roc)[3]) %>% 
    dplyr::arrange(Sensitivity,
                   `1 - Specificity`)
}


# prec_recall.df
## Creates a data frame with TPR (Recall) and Precision (positive predictive value).
## Computes the F1 statistics and uses it to define the optimal threshold (maximum F1)
# dat: dataframe containing predicted probabilities and observed event in two seperate columns
# pred_name: name of the column containing the predicter probabilities
# obs_name: name of the column containing the observed event (factor, with event as the first level)
prec_recall.df <- function(dat,
                           pred_name,
                           obs_name){
  obs <- dat[[obs_name]]
  pred <- dat[[pred_name]]
  
  pred_mod <- prediction(c(pred), obs)
  
  # Precision and recall
  prec_recall <- performance(pred_mod, 'prec', 'rec')
  
  # AUC PR curve
  prauc <- performance(pred_mod, 'aucpr')@y.values[[1]]
  
  #CI prAUC
  n_hat <- log(prauc/(1-prauc))
  n <- min(table(obs))
  tau_hat <- (n * prauc * (1 - prauc))^-.5
  
  
  prauclower <- exp(n_hat - 1.96 * tau_hat) / (1 + exp(n_hat - 1.96 * tau_hat))
  praucupper <- exp(n_hat + 1.96 * tau_hat) / (1 + exp(n_hat + 1.96 * tau_hat))
  
  
  prec_recall_df <- tibble(Threshold = prec_recall@alpha.values[[1]],
                           Precision = prec_recall@y.values[[1]],
                           Recall = prec_recall@x.values[[1]],
                           prAUC = prauc,
                           prAUCLower = prauclower,
                           prAUCUpper = praucupper,
                           F1 = 2 * (Precision * Recall)/(Precision + Recall),
                           optimal_threshold = F1 == max(F1, na.rm = T))
}

# graph.auc
# Returns a AUC (or Precision-Recall) curve.
# Can return superimposed curves for several models (initial df should contains a Model column)
# Can also plot the optimal threshold (df should contain)
# dat: dataframe returned by roc.df or prec_recall.df
# x
# y
# model: name of the column containing the model identifier. If NULL, only one curve is returned
# threshold: name of the column containing the values of the different thresholds
# null.model. type of no_skill model to be plotted ('diag' or 'null':  slope = 1 or slope = 0)
graph.auc <- function(dat,
                      x,
                      y,
                      model = NULL,
                      threshold = 'Threshold',
                      no_skill = 'diag'){
  slope <- ifelse(no_skill == 'diag', 1, 0)
  
  if(is.null(model))
    p <- ggplot(dat, aes(x = !!sym(x),
                         y = !!sym(y)))
  
  if(!is.null(model))
    p <- ggplot(dat, aes(x = !!sym(x),
                         y = !!sym(y),
                         color = !!sym(model)))
  p <- p + 
    geom_line() +
    geom_abline(slope = slope, 
                intercept = 0,
                color = 'gray70') +
    theme_classic()
  
  if(!is.null(threshold)){
    df_threshold <- dat %>%
      dplyr::filter(optimal_threshold) %>% 
      dplyr::mutate_at(threshold, ~ round(., 3))
    
    p <- p +
      geom_point(data = df_threshold,
                 aes(x = !!sym(x),
                     y = !!sym(y)))
    
    return(p)
  }
  return(p)
}


# Takes a list of models and compute various metrics to compare them:
## Only for the training dataset
# AIC for GLMs
# Variable importance 
# 
## For both the training and the test dataset
# AUC
# ROC curves
# Precision_recall curves
# Optimal cutoff threshold
# Confusion matrix
# 
# trained_list = list of model trained using the caret framework

# data = data to use for the prediction(testSet). If data=NULL, the CV data are used

# trainingData = data that were used to train the model. Used to compute the Platt scaling.

# threshold = cut threshold for the hard classification. If nothing is provided, the threshold are computed to optimize F1.

# y_name: name of the predicted variable. Ignored if data=NULL

# discrimination_metrics: character string indicating which discrimination metrics 
# should be plotted (and printed as a table). Should be at least one of the following:
#              "AUC", "prAUC", "Accuracy", "Kappa","Sensitivity", "Specificity", "Pos Pred Value", 
#              "Neg Pred Value", "Precision", "Recall", "F1", "Prevalence", "Detection Rate", 
#              "Detection Prevalence", "Balanced Accuracy", "TPR", "TNR", "FPR", "FNR"

model_metrics <- function(trained_list,
                          data = NULL,
                          trainingData = NULL,
                          threshold = NULL,
                          y_name,
                          discrimination_metrics = c('AUC'),
                          ...){
  
  if(is.null(data)){
    warning('No newdata were provided, the metrics from the cross-validation process will be returned.')
    
    if(any(sapply(trained_list, function(mod) nrow(mod$pred)) == 0))
      stop('No predictions were saved during the CV process!')  
  }
  
  if(is.null(data) & !is.null(trainingData)){
    data <- trainingData
    warning('The metrics computed from the training process will be returned')
  }
  
  
  
  # Variable importance
  ## 0 means that the variable was not selected, NA means that the variable 
  ## was not included in the original model
  varimp_df <- lapply(names(trained_list), function(mod_name){
    mod <- trained_list[[mod_name]]
    
    if(class(try(varImp(mod), silent = T)) == 'try-error'){
      varimp <- tibble(Variable = 'Available variable importance method:',
                       Value = 'No') %>%
        dplyr::rename_at(.vars = 'Value', .funs = ~ mod_name)
    }else{
      
      
      varimp <- varImp(mod)$importance %>%
        rownames_to_column(var = 'Variable') %>%
        dplyr::mutate_all(~ replace_na(., 0))
      names(varimp)[2] <- mod_name
      varimp <- varimp %>% 
        dplyr::mutate_at(mod_name, ~ round(., 3))
    }
    return(varimp)
  }) %>%
    plyr::join_all(type = 'full', by = 'Variable')
  
  if(!is.null(data))
    varimp_df <- NULL
  
  # predicted and observed values for train data (internal calibration)
  data_list <- sapply(names(trained_list), function(mod_name){
    mod <- trained_list[[mod_name]]
    
    # If no newdata frame is included, uses the prediction dataset
    if(is.null(data)){
      pred_data <- mod$pred %>% 
        as_tibble() %>% 
        dplyr::select(rowIndex,
                      Resample,
                      .outcome = obs,
                      .pred = yes)
      return(pred_data)
    }
    
    test_pred <- predict(mod, data, 'prob', na.action = na.pass)$yes
    
    pred_data <- data %>%
      dplyr::rename_at(vars(contains(y_name)), ~ '.outcome') %>% 
      dplyr::mutate('.pred' = test_pred)
    
    
    return(pred_data)
  },
  simplify = F)
  

  # ROC 
  roc_df <- lapply(names(data_list), function(mod_name){
    data <- data_list[[mod_name]]
    
    roc.df(dat = data,
           pred_name = '.pred',
           obs_name = '.outcome') %>% 
      dplyr::mutate(Model = mod_name)
  }) %>% 
    bind_rows() %>% 
    dplyr::mutate(Model = factor(Model,
                                 levels = names(data_list)))
  
  # ROC plots
  p_roc <- graph.auc(dat = roc_df,
                     x = '1 - Specificity', 
                     y = 'Sensitivity', 
                     model = 'Model', 
                     threshold = 'Threshold')
  
  # Precision-recall curve for train and test data
  prec_rec_df <- lapply(names(data_list), function(mod_name){
    data <- data_list[[mod_name]]
    
    prec_recall.df(dat = data,
                   pred_name = '.pred',
                   obs_name = '.outcome') %>% 
      dplyr::mutate(Model = mod_name)
  }) %>% 
    bind_rows() %>% 
    dplyr::mutate(Model = factor(Model,
                                 levels = names(data_list)))
  
  # Create a threshold list
  if(is.null(threshold)){
    # Extract the optimal threshold
    threshold_list <- sapply(names(data_list), function(mod_name){
      prec_rec_df %>% 
        dplyr::filter(Model == mod_name,
                      optimal_threshold == TRUE) %>% 
        pull(Threshold)
    },
    USE.NAMES = T,
    simplify = F)
  }else{
    threshold_list <- as.list(threshold)
  }
  
  # Add threshold to the data
  data_list <- sapply(names(data_list), function(mod_name){
    data_list[[mod_name]] %>% 
      dplyr::mutate(threshold = threshold_list[[mod_name]])
  },
  USE.NAMES = T,
  simplify = F)
  
  # Precision recall plot
  p_prec_recall <- graph.auc(dat = prec_rec_df,
                             x = 'Recall', 
                             y = 'Precision', 
                             model = 'Model', 
                             threshold = 'Threshold',
                             no_skill = 'null')
  
  
  # Add AUC and AIC to the results dataframe
  auc_df <- roc_df %>% 
    dplyr::mutate(AUC = round(AUC, 3),
                  AUCLower = round(AUCLower, 3),
                  AUCUpper = round(AUCUpper, 3)) %>% 
    distinct(Model, AUC, AUCLower, AUCUpper) %>% 
    left_join(prec_rec_df %>% 
                distinct(prAUC, 
                         prAUCLower,
                         prAUCUpper,
                         Model) %>% 
                dplyr::mutate(prAUC = round(prAUC, 3),
                              prAUCLower = round(prAUCLower, 3),
                              prAUCUpper = round(prAUCUpper, 3)),
              by = 'Model') %>% 
    pivot_longer(cols = c(AUC, AUCLower, AUCUpper, prAUC, prAUCLower, prAUCUpper),
                 names_to = 'Variable') %>%
    pivot_wider(values_from = value, 
                names_from = Model)
  
  if(is.null(data)){
    aics <- lapply(names(trained_list), function(name_mod){
      mod <- trained_list[[name_mod]]
      aic <- NULL
      if(is.null(aic))
        aic <- NA_real_
      
      aic <- round(aic, 1)
      
      tibble(AIC = aic) %>% 
        dplyr::mutate(Variable = 'AIC',
                      Model = name_mod)
    }) %>% 
      bind_rows() %>% 
      pivot_wider(values_from = AIC, names_from = Model) 
  }else{
    aics <- NULL
  }
  
  # Creating a confusion matrix
  confusion_list <- sapply(names(data_list), function(name_mod){
    data <- data_list[[name_mod]]
    
    data <- data %>% 
      dplyr::mutate(`.pred` = `.pred` >= threshold_list[[name_mod]]) %>% 
      dplyr::mutate(`.pred` = if_else(`.pred`,
                                      'yes',
                                      'no')) %>% 
      dplyr::mutate(`.pred` = factor(`.pred`, levels = c('no', 'yes')))
    
    pred <- data[['.pred']]
    obs <- data[['.outcome']]
    
    caret::confusionMatrix(data = pred, reference = obs,positive = "yes")
  },
  USE.NAMES = T,
  simplify = F)
  
  
  # Discrimination metrics (AUC, F1, Negative Predictive Value, Precision, Recall, Specificity)
  metrics_table <- auc_df %>% 
    dplyr::rename(Metric = Variable) %>% 
    bind_rows(lapply(names(confusion_list), function(mod_name){
      conf_matrix <- confusion_list[[mod_name]]
      
      metrics <- c(conf_matrix$overall,
                   conf_matrix$byClass)
      
      tibble(Metric = names(metrics),
             Value = metrics) %>% 
        dplyr::rename_at(.vars = 'Value',.funs =  ~ mod_name)
    }) %>% 
      plyr::join_all(by = 'Metric')) %>% 
    dplyr::filter(Metric %in% discrimination_metrics) %>% 
    dplyr::mutate(Metric = factor(Metric,
                                  levels = discrimination_metrics))
  
  # Discrimination metrics graphs
  ## Definition of a color scheme, so that the same color is always assigned to the same metric
  color_table <- tibble(
    Metric = c("AUC", "prAUC", "Accuracy", "Kappa",
               "Sensitivity", "Specificity", "Pos Pred Value", 
               "Neg Pred Value", "Precision", "Recall",
               "F1", "Prevalence", "Detection Rate",
               "Detection Prevalence", "Balanced Accuracy",
               "TPR", "TNR", "FPR", "FNR")) %>% 
    dplyr::mutate(Color = colorRampPalette(RColorBrewer::brewer.pal(8,
                                                                    'Accent'))(n())) %>% 
    dplyr::filter(Metric %in% discrimination_metrics) %>% 
    dplyr::mutate(Metric = factor(Metric,
                                  levels = discrimination_metrics)) %>% 
    dplyr::arrange(Metric)
  
  metrics_plot <- metrics_table %>% 
    dplyr::filter(!(grepl('Upper|Lower', Metric))) %>% 
    pivot_longer(-Metric, 
                 values_to = 'Value', 
                 names_to = 'Model') %>% 
    dplyr::mutate(Model = factor(Model,
                                 names(trained_list))) %>% 
    ggplot(., aes(x = Model, y = Value, fill = Metric)) +
    geom_bar(stat = 'identity',
             position = 'dodge')+
    scale_fill_manual(values = color_table$Color) +
    theme(axis.text.x = element_text(angle = 45,
                                     h = 1))
  
  # Save the results as a list
  results <- list(metrics = varimp_df,
                  plots = list(ROC = p_roc,
                               Precision_recall = p_prec_recall),
                  confusion_matrices = confusion_list,
                  discrimination_metric = metrics_table,
                  discrimination_plot = metrics_plot,
                  predictions = data_list)
  
  return(results)
}

