# Data Science ML Test

### Dataset - Patient Survival Dataset
1. Download data sets and models.
2. Prepare Folder structure as given in the repository.
3. Run train.py file. (optional)
4. Run predict.py file with required arguments.

#### Download Dataset from below link
```bash
https://drive.google.com/drive/folders/1Tic7uw-PeMOrCzchd1GIyp6OamXK7bHT?usp=sharing
```

### Data Description
In this dataset, there are various factors given, which are involved when a patient is hospitalized. On the basis of these factors, predict whether the patient will survive or not.


### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
pip install numpy scipy pandas matplolib seaborn imblearn sklearn xgboost tensorflow
```


### Exploratory Data Analysis

The EDA part is as follows. 
  1. We are reading dataset first.
  2. The data explortation has been done using 
  ```bash
  data.shape #Shape will fetch the tuple of dataframe
  ```
---

  ```bash
    data.info() # information about the DataFrame
  ```
    
--- 
   4. Checking % of missing values
      4.1 The data set have ~3.7% missing values.
   5. Dropping the Columns using drop
   6. Visualizing the death ratio
   ![image](https://github.com/Tecblic-Shubham/DataScience_ML_Test/blob/main/images/download.png)
   7. Visualizing correlation between variables 
   ![image](https://github.com/Tecblic-Shubham/DataScience_ML_Test/blob/main/images/download%20(1).png)
   8. EDA - Using the data like data head, missing values, description, describe, and count. Adding logarithmic scale to the bar graphs for better visibility of the         results.
   9. Preparing two lists and saving them as csv files for modeling.


```bash
python train.py
```

Train File consists Data Loading , Preporcessing and the model code.

### Models used
  1. Logistic Regression
  2. RandomForest
  3. ExtraTrees
  4. AdaBoost
  5. GradientBoosting
  6. Votingr
  7. Dummy
  8. Bagging
  9. XGBoost
  10. LinearSVC
  11. SVC
  12. KNeighbors
  13. DecisionTree
  14. MLP - Multi Layer Perceptron 
  15. Deep Learning squential Model with custom layers

### Inferencing using predict, pass procedure id and will give the answers in terms of success and failure.
```bash
python predict.py --p_id <givetheprocedureidformdatahere>
```` 

### Custom Library as per problem description
```bash
cms_procedures.py
```
This file will contain custom libaray as asked in the problem description. Contains three functions as below.
  1. get_procedure_attributes
  2. get_procedure_success
  3. get_procedure_outcomes


