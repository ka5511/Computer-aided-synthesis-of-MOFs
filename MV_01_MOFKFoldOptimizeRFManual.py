# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:28:39 2022

@author: khali
"""
# Purpose: Experimenting with how to optimize the min_sample_split hyperparameter for the random forest algorithm manually 
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
def cross_validation(model, _X, _y, _cv):
    '''Function to perform x Folds Cross-Validation
        Parameters
        ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
            This is the matrix of features.
      _y: array
            This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
        Returns
        -------
        The function returns a dictionary containing the metrics 'accuracy', 'precision',
        'recall', 'f1' for both training set and validation set.
      '''
    _scoring = ['balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    results = cross_validate(estimator=model, X=_X, y=_y,cv=_cv,scoring=_scoring,return_train_score=True,error_score='raise')
    mat_score = make_scorer(matthews_corrcoef)
    results2 = cross_val_score(estimator=model, X=_X, y=_y,cv=_cv,scoring=mat_score,error_score='raise')
    return {"Training Accuracy scores": results['train_balanced_accuracy'],
              "Mean Training Accuracy": results['train_balanced_accuracy'].mean(),
              "Training Precision scores": results['train_precision_weighted'],
              "Mean Training Precision": results['train_precision_weighted'].mean(),
              "Training Recall scores": results['train_recall_weighted'],
              "Mean Training Recall": results['train_recall_weighted'].mean(),
              "Training F1 scores": results['train_f1_weighted'],
              "Mean Training F1 Score": results['train_f1_weighted'].mean(),
              "Validation Accuracy scores": results['test_balanced_accuracy'],
              "Mean Validation Accuracy": results['test_balanced_accuracy'].mean(),
              "Validation Precision scores": results['test_precision_weighted'],
              "Mean Validation Precision": results['test_precision_weighted'].mean(),
              "Validation Recall scores": results['test_recall_weighted'],
              "Mean Validation Recall": results['test_recall_weighted'].mean(),
              "Validation F1 scores": results['test_f1_weighted'],
              "Mean Validation F1 Score": results['test_f1_weighted'].mean(),
              "MCC": results2
              }



# Bar Charts for both training and testing data
def plot_result(x_label, y_label, plot_title, train_data, val_data,mean_valid):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
          Parameters
          ----------
          x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
          y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
          plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
         
          train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.
        
          val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
          Returns
          -------
          The function returns a Grouped Barchart showing the training and validation result
          in each fold.
        '''
        
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = []
        for i in range(len(train_data)):
            if i == 0:
                labels.append(str(i+1)+'st Fold')
            elif i == 1:
                labels.append(str(i+1)+'nd Fold')
            elif i == 2:
                labels.append(str(i+1)+'rd Fold')
            else:
                labels.append(str(i+1)+'th Fold')
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.00000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Testing')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        mean_value = mean_valid
        text = f"$\: \: Mean \: Testing \: Score = {mean_value:0.3f}$"
        plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
                        fontsize=14, verticalalignment='top',backgroundcolor='0.75')
        plt.show()

# Bar Charts for both training and testing data to plot MCC only
def plot_result_MCC(x_label, y_label, plot_title, val_data,mean_valid):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
          Parameters
          ----------
          x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
          y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
          plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
             
          val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
          Returns
          -------
          The function returns a Grouped Barchart showing the validation result
          in each fold.
        '''
        
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = []
        for i in range(len(val_data)):
            if i == 0:
                labels.append(str(i+1)+'st Fold')
            elif i == 1:
                labels.append(str(i+1)+'nd Fold')
            elif i == 2:
                labels.append(str(i+1)+'rd Fold')
            else:
                labels.append(str(i+1)+'th Fold')
        X_axis = np.arange(len(labels))
        plt.ylim(0.00000, 1)
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Testing')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        mean_value = mean_valid
        text = f"$\: \: Mean \: Testing \: Score = {mean_value:0.3f}$"
        plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
                        fontsize=14, verticalalignment='top',backgroundcolor='0.75')
        plt.show()



# Use Random Forest classifier as the model for cross validation for MOF Shape
splits = 2
accuracies2 = []
mccs2 = []
n_estimator = np.linspace(10, 70, num=60, endpoint=True, retstep=False, dtype=np.int, axis=0)
for i in n_estimator:
    K_clf_model2 = RandomForestClassifier(criterion="entropy",min_samples_split=6,random_state=0,n_estimators=i)
    K_clf_model2_result = cross_validation(K_clf_model2, principal_components, encoded_y3, KFold(n_splits=splits, shuffle=True,random_state=1))
    accuracies2.append(K_clf_model2_result["Mean Validation Accuracy"])
    mccs2.append(K_clf_model2_result["MCC"].mean())
                      
splits = 3
accuracies3 = []
mccs3 = []
for i in n_estimator:
    K_clf_model2 = RandomForestClassifier(criterion="entropy",min_samples_split=6,random_state=0,n_estimators=i)
    K_clf_model2_result = cross_validation(K_clf_model2, principal_components, encoded_y3, KFold(n_splits=splits, shuffle=True,random_state=1))
    accuracies3.append(K_clf_model2_result["Mean Validation Accuracy"])
    mccs3.append(K_clf_model2_result["MCC"].mean())
                      
splits = 4
accuracies4 = []
mccs4 = []
for i in n_estimator:
    K_clf_model2 = RandomForestClassifier(criterion="entropy",min_samples_split=6,random_state=0,n_estimators=i)
    K_clf_model2_result = cross_validation(K_clf_model2, principal_components, encoded_y3, KFold(n_splits=splits, shuffle=True,random_state=1))
    accuracies4.append(K_clf_model2_result["Mean Validation Accuracy"])
    mccs4.append(K_clf_model2_result["MCC"].mean())
            
splits = 5
accuracies5 = []
mccs5 = []
for i in n_estimator:
    K_clf_model2 = RandomForestClassifier(criterion="entropy",min_samples_split=6,random_state=0,n_estimators=i)
    K_clf_model2_result = cross_validation(K_clf_model2, principal_components, encoded_y3, KFold(n_splits=splits, shuffle=True,random_state=1))
    accuracies5.append(K_clf_model2_result["Mean Validation Accuracy"])
    mccs5.append(K_clf_model2_result["MCC"].mean())
            
plt.plot(n_estimator,accuracies2, scalex=True, scaley=True, data=None, marker='o',color='b',label="2 K-Fold")
plt.plot(n_estimator,accuracies3, scalex=True, scaley=True, data=None, marker='x',color='r',label="3 K-Fold")
plt.plot(n_estimator,accuracies4, scalex=True, scaley=True, data=None, marker='1',color='k',label="4 K-Fold")
plt.plot(n_estimator,accuracies5, scalex=True, scaley=True, data=None, marker='s',color='g',label="5 K-Fold")
plt.title("Number of Forest Trees vs Accuracy for MOF Quality", fontsize=15)
plt.xlabel("Number of Forest Trees", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend()
plt.grid()
plt.show()


plt.plot(n_estimator,mccs2, scalex=True, scaley=True, data=None, marker='o',color='b',label="2 K-Fold")
plt.plot(n_estimator,mccs3, scalex=True, scaley=True, data=None, marker='x',color='r',label="3 K-Fold")
plt.plot(n_estimator,mccs4, scalex=True, scaley=True, data=None, marker='1',color='k',label="4 K-Fold")
plt.plot(n_estimator,mccs5, scalex=True, scaley=True, data=None, marker='s',color='g',label="5 K-Fold")
plt.title("Number of Forest Trees vs MCC for MOF Quality", fontsize=15)
plt.xlabel("Number of Forest Trees", fontsize=14)
plt.ylabel("MCC", fontsize=14)
plt.legend()
plt.grid()
plt.show()
