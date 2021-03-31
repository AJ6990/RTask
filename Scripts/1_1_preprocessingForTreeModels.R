# Data preparation 
# make training and test sets ready for tree-based models (they can handle categorical variables
# and are not sensitive to numerical scales)

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
  

# 03. Divide data into train/test with a split ratio of 75::25
set.seed(888) # should be the same for all, so that the training/test sets are all the same
splitData <- createDataPartition(InputData$label, p = 0.75, list = FALSE)

trainingData<- InputData[splitData,]
testData<- InputData[-splitData,]

# 04. Extract first digit of zip code as broad area and first 3 digits for county information
# Weight of evidence encoding is used to convert categorical variables (zip codes)
# with high cardinallity to numeric values
trainingData <- trainingData %>% 
  dplyr::mutate(zip1 = as.factor(substr(zip.code,1,1))) %>% # First digit of zip code
  dplyr::mutate(zip3 = as.factor(substr(zip.code,1,3))) %>% # First 3 digits of zip code
  dplyr::mutate(woe1 = round(WOE(zip1,label),3)) %>%  # new variable: weight of evidence for zip1 
  dplyr::mutate(woe3 = round(WOE(zip3,label),3)) # weight of evidence of zip3
  

# 05. Detect outliers in two new calculated variables 
outlierWoe1 <- boxplot.stats(trainingData$woe1)$out # Outliers of new variable woe1
outlierWoe3 <- boxplot.stats(trainingData$woe3)$out # Outliers of new variable woe3

# 06. Winsorizing Outlier for woe1 and woe3: replace the extreme outliers with maximum values at threshold
trainingData <- trainingData %>% 
  dplyr::mutate(woe1_win = Winsorize(woe1,probs = c(0.15, 0.85),type = 1)) %>% # 85% winsorization
  dplyr::mutate(woe3_win = Winsorize(woe3,probs = c(0.05, 0.95),type = 1)) %>% # 95% winsorization
  dplyr::select(-c(woe1,woe3,zip.code)) %>% 
  dplyr::rename(broadArea = woe1_win,
         county = woe3_win)

# 07. Handling no values of the Sports column 
## Define a new value: Other
levels(trainingData$sports)[match("",levels(trainingData$sports))] <- "other"
trainingData$sports[trainingData$sports == ""] <- factor("other")


# 08. repeat the same process of extracting information from Zip Code but in Test set
testData <- testData %>% 
  dplyr::mutate(zip1 = as.factor(substr(zip.code,1,1))) %>% 
  dplyr::mutate(zip3 = as.factor(substr(zip.code,1,3))) %>%
  select(-zip.code)

## 09. Transformation of zip codes into numeric values using the results of Training set.
testData <- testData %>% 
  left_join(x = ., y = trainingData %>%
            select("zip1","broadArea") %>%
            unique(),
          by = "zip1") %>% 
  left_join(x = ., y = trainingData %>%
            select("zip3","county") %>%
            unique(),
          by = "zip3")
  

# 10. Handling sports variable in Test set
# Like in training set, a new variable as "other" is defined
levels(testData$sports)[match("",levels(testData$sports))] <- "other"
testData$sports[testData$sports == ""] <- factor("other")


# 11. Removing some of the original variables, so that the variables of interest are only appeared in training/test sets 
trainingData <- trainingData %>%
  select(-c(name, zip1,zip3)) %>%
  dplyr::mutate(label = ifelse(label == 0,"no","yes")) %>% 
  dplyr::mutate(label = as.factor(label))

testData <- testData %>%
  select(-c(name, zip1,zip3)) %>%
  dplyr::mutate(label = ifelse(label == 0,"no","yes")) %>% 
  dplyr::mutate(label = as.factor(label))


formula_list <- list('advertisementPrediction' = formula(paste('label ~ .')))

