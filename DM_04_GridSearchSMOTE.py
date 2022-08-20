# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:28:39 2022

@author: Khalid Alotaibi
"""
# Purpose: Perform exhaustive grid search optimization over specified parameter values using stratified K-fold for cross-validation. If data augmentation is applied, the applied technique is reflected on the file title
####################################################################

# Step 1: Import needed libraries   
import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from numpy import std
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
import joblib
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline

# Step 2: Import Dataset
RandForDataset = pandas.read_csv("UTSA_16_A2ML_Data_Updated.csv")
condlist = ['Zinc Acetate Zn(CH3CO2)2 Mass (g)', 
            'Citric Acid HOC(CO2H)(CH2CO2H)2 Mass (g)', 
            'Potassium Hydroxide KOH Mass (g)', 'Water Volume (ml)', 
            'Ethanol EtOH Volume (ml)', 'Synthesis Time (hr)']

condlist_znco = ['Cobalt Acetate Co(C2H3O2)2 Mass (g)','Zinc Acetate Zn(CH3CO2)2 Mass (g)', 
            'Citric Acid HOC(CO2H)(CH2CO2H)2 Mass (g)', 
            'Potassium Hydroxide KOH Mass (g)', 'Water Volume (ml)', 
            'Ethanol EtOH Volume (ml)', 'Synthesis Time (hr)']

X = np.array(RandForDataset[condlist_znco])
Y1 = np.array(RandForDataset['Monolith Transparency\nClear - Glass-looking\nCloudy - White-colored']);
Y2 = np.array(RandForDataset['Monolith?']);
Y3 = [0]*len(Y2)
number_of_datapoints = len(Y1)
for i in range(number_of_datapoints):
    if Y1[i] == 'Clear' and Y2[i] == 'Y':
        Y3[i] = 'Transparent Monolith'
    elif Y1[i] == 'Cloudy' and Y2[i] == 'Y':
        Y3[i] = 'White-colored Monolith'
    elif Y2[i] == 'N':
        Y3[i] = 'Powder'
        
# Define the number of variables/properties/features
number_of_features = len(X[0])

# performing preprocessing part to scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)

principal_components = pca.fit_transform(X_scaled)



# Convert target variables from 'str' to 'int'
from sklearn.preprocessing import LabelEncoder
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()
encoded_y1 = label_encoder1.fit_transform(Y1)
encoded_y2 = label_encoder2.fit_transform(Y2)
encoded_y3 = label_encoder3.fit_transform(Y3)
label_encoder_name_mapping1 = dict(zip(label_encoder1.classes_,label_encoder1.transform(label_encoder1.classes_)))
label_encoder_name_mapping2 = dict(zip(label_encoder2.classes_,label_encoder2.transform(label_encoder2.classes_)))
label_encoder_name_mapping3 = dict(zip(label_encoder3.classes_,label_encoder3.transform(label_encoder3.classes_)))
print("Mapping of Transparency Encoded Classes", label_encoder_name_mapping1)
print("Encoded Transparency Values",encoded_y1)
print("Mapping of MOF Shape Encoded Classes", label_encoder_name_mapping2)
print("Encoded MOF Shape Values",encoded_y2)
print("Mapping of MOF Quality Encoded Classes", label_encoder_name_mapping3)
print("Encoded MOF Quality Values",encoded_y3)

# K-fold cross-validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

# Use paramaters obtained from Randomized search with 20 iteration then 100 iterations 
from sklearn.model_selection import GridSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1100, stop = 1800, num = 7)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 40, num = 4)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [False, True]
# First create the base model to tune
rf = RandomForestClassifier(random_state=1)
# Define a pipeline
steps = [('over', BorderlineSMOTE(k_neighbors=1)), ('model', rf)]
pipeline = Pipeline(steps=steps)
# Create the random grid
random_grid = {'model__n_estimators': n_estimators,
                'model__max_depth': max_depth,
                'model__min_samples_split': min_samples_split,
                'model__min_samples_leaf': min_samples_leaf,
                'model__bootstrap': bootstrap}

# Number of folds to be used
splits = 3
# Use the random grid to search for best hyperparameters for MOF Shape
# Create the training & testing sets
train_features2, test_features2, train_labels2, test_labels2 = train_test_split(principal_components,encoded_y2, test_size = .20, random_state= 1)
train_features3, test_features3, train_labels3, test_labels3 = train_test_split(principal_components,encoded_y3, test_size = .20, random_state= 1)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_2_random = GridSearchCV(estimator = pipeline, param_grid = random_grid, cv = splits, scoring=make_scorer(matthews_corrcoef), n_jobs = None, error_score='raise')
# Fit the random search model
best_grid_model_2 = rf_2_random.fit(train_features2, train_labels2)

filename2 = "SMOTE_RF_Grid_Optimized_Shape.joblib"
joblib.dump(best_grid_model_2, filename2)

# Use the random grid to search for best hyperparameters for MOF Quality
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf = RandomForestClassifier(random_state=1)
# Define a pipeline
steps = [('over', BorderlineSMOTE(k_neighbors=1)), ('model', rf)]
pipeline = Pipeline(steps=steps)

rf_3_random = GridSearchCV(estimator = pipeline, param_grid = random_grid, cv = splits, scoring=make_scorer(matthews_corrcoef), n_jobs = None, error_score='raise')
# Fit the random search model
best_grid_model_3 = rf_3_random.fit(train_features3, train_labels3)

filename3 = "SMOTE_RF_Grid_Optimized_Quality.joblib"
joblib.dump(best_grid_model_3, filename3)
