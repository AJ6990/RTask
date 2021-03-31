Project folder contains four subfolders:
1- Data: dataset + trained models
2- Scripts: all scripts for running the project
3- Results: An Rmarkdown html output
4- Functions: two functions made for glm model weight
optimization and the other one for performance metrics'
evaluation



Scripts folder:
1- Each step are named by number (subnumbers) and a general information about the content. 
2- First 4 scripts are used to extract and prepare the data. The running time are short. 
3- The models are trained with the scripts that starts with 2_x. Since they are trained and cross validated by hyper-grid search, some of them (SVM, RF, GBM) takes a bit longer to run. 
4- The script 4 relies on helper functions that you can find in Function folder.  The actual extraction of information from the model only starts line 163, the first part is only there to create basic statistics for the report. 

5- The up to date version of the report script is 5_Case_Study_Project.Rmd. By running this file, the report is automatically generated in results folder. 

### Project path should be changed if you would like to run it on your device (or at least to be matched)

### cases are defined within each preprocessing step.