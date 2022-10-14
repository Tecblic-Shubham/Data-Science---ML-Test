# Importing required Libraries
import cms_procedures

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import os
import joblib
from sklearn import metrics
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier 
from xgboost.sklearn import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2

# LOAD DATA
data = cms_procedures.get_data()
cat_col = pd.read_csv('./dataset/cat_col.csv')['0'].tolist()
num_col = pd.read_csv('./dataset/num_col.csv')['0'].tolist()

# Preprocess data
preprocess_pipeline = ColumnTransformer(transformers=
                                        [('num', SimpleImputer(strategy='median'),num_col),
                                        ('cat',OneHotEncoder(),cat_col)]
                                       )

y = data.hospital_death
X = data.drop('hospital_death',axis=1)

scaler = StandardScaler()
X_transform = preprocess_pipeline.fit_transform(X)
X_scaled = scaler.fit_transform(X_transform)
joblib.dump(scaler, "./Models/Scaler.joblib")

pca=PCA()
pca.fit_transform(X_scaled)
explained_variance_s=pca.explained_variance_ratio_
joblib.dump(pca, "./Models/PCA.joblib")

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)

# Defining Models
estimators = [
    ('lr',LogisticRegression()),
    ('dtc',DecisionTreeClassifier()),    
    ('rfc',RandomForestClassifier()),
    ('knc',KNeighborsClassifier())]

classifier_models = [
    LogisticRegression(max_iter=500),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    VotingClassifier(estimators=estimators),
    DummyClassifier(strategy='stratified'),
    BaggingClassifier(DecisionTreeClassifier(),
                                  max_samples=0.7,
                                   bootstrap=True),
    XGBClassifier(),
    LinearSVC(),
    SVC(),   
    KNeighborsClassifier(2),
    DecisionTreeClassifier(),
    MLPClassifier((64,32,16),max_iter=1000),
   
]

oversamplers=[None,SMOTE()]

labels= [
    "LR",
    "RandomForest",
    "ExtraTrees",
     "AdaBoost",
     "GradientBoosting",
    "Votingr",
    "Dummy",
    "Bagging",
    "XGB",
    "LinearSVC",
    "SVC",   
    "KNeighbors",
    "DecisionTree",
    "MLP"
        ]

# Training ML Models
score = []
oversampling = []
names = []
precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
time_ = []
i=0
for model in classifier_models:    
    i=i+1
    for over in oversamplers: 
    
        clf = Pipeline(steps=[("preprocessor", preprocess_pipeline),('scaler',StandardScaler()),('over',over),('pca',PCA(n_components=60)),("classifier", model),])
        start_time = time.time()
        clf.fit(X_train, y_train)   
        end_time = time.time()
        names.append(f'{labels[i-1]}')
        oversampling.append(f'{over}')    
        
        precision_score.append(metrics.precision_score(y_test, clf.predict(X_test)))
        recall_score.append(metrics.recall_score(y_test, clf.predict(X_test)))
        f1_score.append( metrics.f1_score(y_test, clf.predict(X_test)))
        accuracy_score.append(metrics.accuracy_score(y_test, clf.predict(X_test)))

        joblib.dump(
        clf,  
        os.path.join("Models", f"{labels[i-1]}_accuracy_{round(accuracy_score[-1],2)}.pkl") 
        )    
        print(f"{model}:{accuracy_score[-1]}")
        time_.append(f'{round(end_time-start_time,2)}s')
            
### Deep Learning ###
# Sequential Models

# Spliting transformed data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.3,stratify=y)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Create custom model
def create_model( layer_1,layer_2,layer_3,layer_4, dropout=0.0,regu=0.0):
    '''
        Create Squential Model with 4 layers with dropout and regularization
    '''
    optimizer='adam'
    init='glorot_uniform'
    reg = l2(regu)
    model = Sequential()   
    model.add(Dense(layer_1,input_dim=X_train.shape[1],activation='relu',kernel_initializer=init))
    model.add(Dense(layer_2,activation='relu',kernel_regularizer=reg,kernel_initializer=init))
    model.add(Dropout(dropout))
    model.add(Dense(layer_3,activation='relu',kernel_regularizer=reg,kernel_initializer=init))
    model.add(Dropout(dropout))
    model.add(Dense(layer_4,activation='relu',kernel_regularizer=reg,kernel_initializer=init))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid',kernel_regularizer=reg,kernel_initializer=init))
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=["accuracy",Recall(),Precision()])
    
    return model

# Define different models
models = [create_model(512,256,128,64,dropout=0.25,regu=0.001),
          create_model(256,128,64,32,dropout=0.25,regu=0.001),
          create_model(128,64,32,24,dropout=0.25,regu=0.001),
          create_model(64,32,16,8,dropout=0.25,regu=0.001),
         ] 

early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Training DL Models 
acc = []
prec= []
f1=[]
recall=[]
model_name=[]
for model in models:    
    model.fit(X_train,y_train,epochs=100,callbacks=[early_stopping],validation_batch_size=0.1,verbose=2)
    model_name.append(model.name)
    x_test = []
    for i in model.predict(X_test):
        if i > 0.5:
            x_test.append(1)
        else:
            x_test.append(0)
    prec.append(metrics.precision_score(y_test, x_test))
    recall.append(metrics.recall_score(y_test,x_test))
    f1.append( metrics.f1_score(y_test, x_test))
    acc.append(metrics.accuracy_score(y_test,x_test))

    # Saving Sequential Model
    save_dir = './Models/'
    saver = save_model(model,f"{save_dir}{model.name}")
    
