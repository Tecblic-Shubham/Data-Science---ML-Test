# Data Science ML Test

### Dataset - Patient Survival Dataset

#### Download Dataset from below link
```bash
https://drive.google.com/drive/folders/1QWdi2WNZbkoYBAw_H0SVy1csWR4TMqqi?usp=sharing
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
  3. ```bash
    data.info() # information about the DataFrame
    ```
   4. Checking % of missing values
      4.1 The data set have ~3.7% missing values.
   5. Dropping the Columns using drop
   6. Visualizing the death ratio
   ![image](https://github.com/Tecblic-Shubham/DataScience_ML_Test/blob/main/images/download.png)
   7. Visualizing correlation between variables 
   ![image](https://github.com/Tecblic-Shubham/DataScience_ML_Test/blob/main/images/download%20(1).png)
   8. EDA - Using the data like data head, missing values, description, describe, and count. Adding logarithmic scale to the bar graphs for better visibility of the         results.
   9. Preparing two lists and saving them as csv files for modeling.


