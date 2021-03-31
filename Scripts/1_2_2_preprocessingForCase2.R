# prepare data (+ training and test sets for logistic regression and support vector machine)
# This script convert categorical variables with 2, 3 and 4 levels using dummy encoding. 
# Categorical variable with high levels are also encoded as weight of evidence approach


library(dplyr)
library(ggplot2)
library(fastDummies)
library(caret)
library(InformationValue)
library(DescTools)
library(dataPreparation)


rm(list = ls())

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
# Weight of evidence encoding is used to convert categorical variables with high cardinallity to numeric values
trainingData <- trainingData %>% 
  dplyr::mutate(zip1 = as.factor(substr(zip.code,1,1))) %>% # First digit of zip code
  dplyr::mutate(zip3 = as.factor(substr(zip.code,1,3))) %>% # First 3 digits of zip code
  dplyr::mutate(woeZip1 = round(WOE(zip1,label),3)) %>%  # weight of evidence of zip1 
  dplyr::mutate(woeZip3 = round(WOE(zip3,label),3)) # weight of evidence of zip3


# 05. Handling no values of Sports column 
## Define a new value: Other, and set the levels
levels(trainingData$sports)[match("",levels(trainingData$sports))] <- "other"
trainingData$sports[trainingData$sports == ""] <- factor("other")

# 06. Handling low cardinality categorical variables
## binary categorical variables such as family.status, car and living.area are coded as 1 and 0
## Categorical variables with 3 or 4 levels are encoded using weight of evidence approach
## To avoid multicollinarity, one of th values within each values are removed.
trainingData <- trainingData %>% 
  dplyr::mutate(familyStatusSingle = ifelse(family.status == "single",1,0)) %>% # 1 as single, 0 as married
  dplyr::mutate(carPractical = ifelse(car == "practical",1,0)) %>% # 1 as practical car, 0 as expensive
  dplyr::mutate(livingAreaRural = ifelse(living.area == "rural",1,0)) %>% # 1 as rural area, 0 as urban
  dplyr::mutate(lifeStyleActive = ifelse(lifestyle == "active",1,0)) %>% # dummy encoded: true values --> 1
  dplyr::mutate(lifeStyleHealthy = ifelse(lifestyle == "healthy",1,0)) %>% # dummy encoded: true values --> 1
  dplyr::mutate(sportsAthletics = ifelse(sports == "athletics",1,0)) %>% # dummy encoded: true values --> 1
  dplyr::mutate(lifeStyleBadminton = ifelse(sports == "badminton",1,0)) %>% # dummy encoded: true values --> 1
  dplyr::mutate(lifeStyleSoccer = ifelse(sports == "soccer",1,0)) %>% # dummy encoded: true values --> 1
  select(-c(family.status,car,living.area,sports,lifestyle))


# 07. Detect outliers in woeZip1 and woeZip3
outlierWoeZip1 <- boxplot.stats(trainingData$woeZip1)$out # Outliers of new variable woe1
outlierWoeZip3 <- boxplot.stats(trainingData$woeZip3)$out # Outliers of new variable woe1

# 08. Winsorizing Outlier for woe1 and woe3: replace the extreme outliers with maximum values at threshold
trainingData <- trainingData %>% 
  dplyr::mutate(woeZip1_win = Winsorize(woeZip1,probs = c(0.15, 0.85),type = 1)) %>% # 85% winsorization
  dplyr::mutate(woeZip3_win = Winsorize(woeZip3,probs = c(0.05, 0.95),type = 1)) %>% # 95% winsorization
  dplyr::select(-c(woeZip1,woeZip3,zip.code)) %>% 
  dplyr::rename(broadArea = woeZip1_win,
         county = woeZip3_win)

# 09. Standardization of numerical variables to have guassian-shape distribution
## find mean and sd column-wise of training data
colInterest <- trainingData %>% 
  select(-c("name","label","zip1","zip3")) %>% names() # interests: numerical variables
trainMean <- apply(trainingData[colInterest],2,mean)
trainSd <- apply(trainingData[colInterest],2,sd)

### centered and scaled
trainingData[colInterest] <- sweep(sweep(trainingData[colInterest], 2L, trainMean),
                                   2, trainSd, "/")


# 10. Handling sports variable in Test set
# Like in training set, a new variable as "other" is defined
levels(testData$sports)[match("",levels(testData$sports))] <- "other"
testData$sports[testData$sports == ""] <- factor("other")

# 11. repeat the same process of extracting information from Zip Code but in Test set
testData <- testData %>% 
  dplyr::mutate(zip1 = as.factor(substr(zip.code,1,1))) %>% 
  dplyr::mutate(zip3 = as.factor(substr(zip.code,1,3))) %>% 
  select(-zip.code)


# 12. Handling low cardinality categorical variables in Test set
## categorical variables with 2, 3 and 4 levels such are encoded as dummy variables
testData <- testData %>% 
  dplyr::mutate(familyStatusSingle = ifelse(family.status == "single",1,0)) %>% 
  dplyr::mutate(carPractical = ifelse(car == "practical",1,0)) %>% 
  dplyr::mutate(livingAreaRural = ifelse(living.area == "rural",1,0)) %>% 
  dplyr::mutate(lifeStyleActive = ifelse(lifestyle == "active",1,0)) %>%
  dplyr::mutate(lifeStyleHealthy = ifelse(lifestyle == "healthy",1,0)) %>%
  dplyr::mutate(sportsAthletics = ifelse(sports == "athletics",1,0)) %>%
  dplyr::mutate(lifeStyleBadminton = ifelse(sports == "badminton",1,0)) %>%
  dplyr::mutate(lifeStyleSoccer = ifelse(sports == "soccer",1,0)) %>%
  select(-c(family.status,car,living.area,sports,lifestyle))


## 13. Transformation of zip codes into numeric values using the results of Training set.
testData <- testData %>% 
  left_join(x = ., y = trainingData %>%
              select("zip1","broadArea") %>%
              unique(),
            by = "zip1") %>% 
  left_join(x = ., y = trainingData %>%
              select("zip3","county") %>%
              unique(),
            by = "zip3")

# 14. centered and scaled Test data using Training mean and SD
testData[colInterest] <- sweep(sweep(testData[colInterest], 2L, trainMean),
                               2, trainSd, "/")

# 15. Removing some of the original variables, so that the variables of interest are only appeared in training/test sets 
trainingData <- trainingData %>%
  select(-c(name, zip1,zip3)) %>%
  dplyr::mutate(label = ifelse(label == 0,"no","yes")) %>% 
  dplyr::mutate(label = as.factor(label))

testData <- testData %>%
  select(-c(name, zip1,zip3)) %>%
  dplyr::mutate(label = ifelse(label == 0,"no","yes")) %>% 
  dplyr::mutate(label = as.factor(label))


formula_list <- list('advertisementPrediction' = formula(paste('label ~ .')))


