# Data preparation to train the whole dataset using  the best model


library(dplyr)
library(ggplot2)
library(fastDummies)
library(caret)
library(InformationValue)
library(DescTools)
library(dataPreparation)


# rm(list = ls())

# Project path
project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'

# 01.Load Data
InputData <-  read.csv(file.path(project_path,'Data/Recruiting_Task_InputData.csv'))

# 02. Factorize the character values and convert target labels to 0 (no response) and 1 (response)
InputData <- dplyr::mutate_if(InputData, is.character, as.factor) %>% 
  dplyr::mutate(label = ifelse(label == "response",1,0))


# 03. Extract first digit of zip code as broad area and first 3 digits for county information
# Weight of evidence encoding is used to convert categorical variables (zip codes)
# with high cardinallity to numeric values
Data <- InputData %>% 
  dplyr::mutate(zip1 = as.factor(substr(zip.code,1,1))) %>% # First digit of zip code
  dplyr::mutate(zip3 = as.factor(substr(zip.code,1,3))) %>% # First 3 digits of zip code
  dplyr::mutate(woe1 = round(WOE(zip1,label),3)) %>%  # new variable: weight of evidence for zip1 
  dplyr::mutate(woe3 = round(WOE(zip3,label),3)) # weight of evidence of zip3


rm(list = "InputData")

# 04. Detect outliers in two new calculated variables 
outlierWoe1 <- boxplot.stats(Data$woe1)$out # Outliers of new variable woe1
outlierWoe3 <- boxplot.stats(Data$woe3)$out # Outliers of new variable woe3

# 06. Winsorizing Outlier for woe1 and woe3: replace the extreme outliers with maximum values at threshold
Data <- Data %>% 
  dplyr::mutate(woe1_win = Winsorize(woe1,probs = c(0.15, 0.85),type = 1)) %>% # 85% winsorization
  dplyr::mutate(woe3_win = Winsorize(woe3,probs = c(0.05, 0.95),type = 1)) %>% # 95% winsorization
  dplyr::select(-c(woe1,woe3)) %>% 
  dplyr::rename(broadArea = woe1_win,
                county = woe3_win)

# 04. Handling no values of the Sports column 
## Define a new value: Other
levels(Data$sports)[match("",levels(Data$sports))] <- "other"
Data$sports[Data$sports == ""] <- factor("other")


# 05. Removing some of the original variables, so that the variables of interest are only appeared in training/test sets 
Data <- Data %>%
  select(-c(name, zip.code)) %>%
  dplyr::mutate(label = ifelse(label == 0,"no","yes")) %>% 
  dplyr::mutate(label = as.factor(label))

