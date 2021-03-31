project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'
source(file.path(project_path, "Scripts/2_1_train_GLM.R"))
rm(list = ls())
gc()

project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'
source(file.path(project_path, "Scripts/2_2_train_RF.R"))
rm(list = ls())
gc()

project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'
source(file.path(project_path, "Scripts/2_3_train_XGB.R"))
rm(list = ls())
gc()


project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'
source(file.path(project_path, "Scripts/2_4_train_DT.R"))
rm(list = ls())
gc()

project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'
source(file.path(project_path, "Scripts/2_5_train_GBM.R"))
rm(list = ls())
gc()

project_path <- 'C:/Users/Alizadeh/Documents/Altran/Recruiting_Task'
source(file.path(project_path, "Scripts/2_6_train_SVM.R"))
rm(list = ls())
gc()