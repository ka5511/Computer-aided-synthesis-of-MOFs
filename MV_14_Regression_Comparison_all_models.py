# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:54:05 2022

@author: Khalid Alotaibi
"""
# Purpose: Perform LeaveOneOut cross-validation (LOOCV) to check the performance of all selected regression models in predicting quantitative MOF synthesis results 
######################################################
# Step 1: Import needed libraries   
import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model # Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from numpy import std
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut


# Step 2: Import Dataset
RandForDataset_all = pandas.read_csv("UTSA_16_A2ML_Data_Updated.csv")
RandForDataset = RandForDataset_all.dropna()     #drop all rows that have any NaN values

print("The column headers :")
print(list(RandForDataset.columns.values))
Y1 = np.array(RandForDataset['Bulk Density (g/ml)']);

synthesis_features = ['Zinc Acetate Zn(CH3CO2)2 Mass (g)', 
            'Citric Acid HOC(CO2H)(CH2CO2H)2 Mass (g)', 
            'Potassium Hydroxide KOH Mass (g)', 'Water Volume (ml)', 
            'Ethanol EtOH Volume (ml)', 'Synthesis Time (hr)']

synthesis_features_abb = ['Zn Mass', 
            'Acid Mass', 
            'Base Mass', 'Water Volume', 
            'EtOH Volume', 'Synthesis Time']

synthesis_outputs = ['Bulk Density (g/ml)','Average Particle Size',
                     'BET Area (m2/g)','Micropore Volume (cm3/g)',
                     'Pore Volume (cm3/g)']

X = np.array(RandForDataset[synthesis_features])
Y = np.array(RandForDataset[synthesis_outputs])



def get_models_density():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(make_pipeline(StandardScaler(), SVR()))
    return models

def get_models_size():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(make_pipeline(StandardScaler(), SVR()))
    return models

def get_models_BET():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(make_pipeline(StandardScaler(), SVR()))
    return models

def get_models_micro_volume():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(make_pipeline(StandardScaler(), SVR()))
    return models


def get_models_total_volume():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(make_pipeline(StandardScaler(), SVR()))
    return models

models_list = ['RF','LR','Ridge','Lasso','KNN','SVR']
models_list_expanded = ['Random Forest','Linear Regression','Ridge',
                        'Lasso','KNN','Support Vector']
models_outputs = ['Density', 'Particle Size', 'BET Area', 'Micropore Volume','Total Pore Volume']

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
    _scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
    results = cross_validate(estimator=model, X=_X, y=_y,cv=_cv,scoring=_scoring,return_train_score=True,error_score='raise')
    return {"Training Mean Absolute Error": results['train_neg_mean_absolute_error']*(-1),
              "Mean Training Absolute Error": results['train_neg_mean_absolute_error'].mean()*(-1),
              "Training Root Mean Squared Error": results['train_neg_root_mean_squared_error']*(-1),
              "Mean Training Root Mean Squared Error": results['train_neg_root_mean_squared_error'].mean()*(-1),
              "Training R2 Scores": results['train_r2'],
              "Mean Training R2": results['train_r2'].mean(),
              "Validation Mean Absolute Error": results['test_neg_mean_absolute_error']*(-1),
              "Mean Validation Absolute Error": results['test_neg_mean_absolute_error'].mean()*(-1),
              "Validation Root Mean Squared Error": results['test_neg_root_mean_squared_error']*(-1),
              "Mean Validation Root Mean Squared Error": results['test_neg_root_mean_squared_error'].mean()*(-1),
              }


# performing preprocessing part to scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
principal_components = pca.fit_transform(X_scaled)

# Store results of all models for all outputss in lists 
models_results_density = list()
models_results_size = list()
models_results_BET = list()
models_results_micro = list()
models_results_total_volume = list()

def plot_result(x_label, y_label, plot_title, train_data, val_data):
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
        ind = np.arange(len(train_data))
        plt.figure(figsize=(12,6))
        plt.bar(ind, train_data, 0.4, color='blue', label='Training')
        if val_data != None:
            plt.bar(ind+0.4, val_data, 0.4, color='red', label='Testing')
        plt.title(plot_title, fontsize=30)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend(loc='best')
        plt.xticks(ind + 0.4 / 2, (models_list))
        plt.grid(True)
        plt.show()

def repeated_Kfold(model,_x,_y,_output_index,_cv,_splits,_model_index):
    kfold_results = cross_validation(model, _x, _y, _cv)
    # plot_result(models_list_expanded[_model_index]+" LOOCV", "Mean Absolute Error", "MOF "+models_outputs[_output_index]+" Mean Absolute Error in "+str(_splits)+" Folds", kfold_results["Training Mean Absolute Error"], kfold_results["Validation Mean Absolute Error"], kfold_results["Mean Validation Absolute Error"])
    # plot_result(models_list_expanded[_model_index]+" LOOCV", "Root Mean Squared Error", "MOF "+models_outputs[_output_index]+" Root Mean Squared Error in "+str(_splits)+" Folds", kfold_results["Training Root Mean Squared Error"], kfold_results["Validation Root Mean Squared Error"], kfold_results["Mean Validation Root Mean Squared Error"])
    mae_scores = kfold_results["Training Mean Absolute Error"]
    rmse_scores = kfold_results["Training Root Mean Squared Error"]
    r2_scores = kfold_results["Training R2 Scores"]
    mae_scores_testing = kfold_results['Validation Mean Absolute Error']
    rmse_scores_testing = kfold_results['Validation Root Mean Squared Error']
    return mae_scores,rmse_scores,r2_scores, kfold_results, mae_scores_testing,rmse_scores_testing

splits = 'LOOCV'
i = 0
j = 0
models_mae_scores = list()
models_rmse_scores = list()
models_r2_scores = list()
models_all_scores = list()
models_mae_scores_testing = list()
models_rmse_scores_testing = list()
models_mae_mean = list()
models_rmse_mean = list()
models_r2_mean = list()
models_mae_testing_mean = list()
models_rmse_testing_mean= list()

for model in get_models_density():
    a,b,c,d,e,f =  repeated_Kfold(model, X, Y[:,i],0, LeaveOneOut(),splits,j)
    models_mae_scores.append(a)
    models_mae_mean.append(a.mean())
    models_rmse_scores.append(b)
    models_rmse_mean.append(b.mean())
    models_r2_scores.append(c)
    models_r2_mean.append(c.mean())
    models_all_scores.append(d)
    models_mae_scores_testing.append(e)
    models_mae_testing_mean.append(e.mean())
    models_rmse_scores_testing.append(f)
    models_rmse_testing_mean.append(f.mean())
    j=1+j

plot_result('Model Name', 'Mean Absolute Error', models_outputs[i]+' Prediction Performance', models_mae_mean, models_mae_testing_mean)
plot_result('Model Name', 'Root Mean Squared Error', models_outputs[i]+' Prediction Performancee', models_rmse_mean, models_rmse_testing_mean)
plot_result('Model Name', 'R\u00b2', models_outputs[i]+' Prediction Performance', models_r2_mean, None)
i = i+1

j = 0
models_mae_scores = list()
models_rmse_scores = list()
models_r2_scores = list()
models_all_scores = list()
models_mae_scores_testing = list()
models_rmse_scores_testing = list()
models_mae_mean = list()
models_rmse_mean = list()
models_r2_mean = list()
models_mae_testing_mean = list()
models_rmse_testing_mean= list()

for model in get_models_size():
    a,b,c,d,e,f =  repeated_Kfold(model, X, Y[:,i],0, LeaveOneOut(),splits,j)
    models_mae_scores.append(a)
    models_mae_mean.append(a.mean())
    models_rmse_scores.append(b)
    models_rmse_mean.append(b.mean())
    models_r2_scores.append(c)
    models_r2_mean.append(c.mean())
    models_all_scores.append(d)
    models_mae_scores_testing.append(e)
    models_mae_testing_mean.append(e.mean())
    models_rmse_scores_testing.append(f)
    models_rmse_testing_mean.append(f.mean())
    j=1+j

plot_result('Model Name', 'Mean Absolute Error', models_outputs[i]+' Prediction Performance', models_mae_mean, models_mae_testing_mean)
plot_result('Model Name', 'Root Mean Squared Error', models_outputs[i]+' Prediction Performancee', models_rmse_mean, models_rmse_testing_mean)
plot_result('Model Name', 'R\u00b2', models_outputs[i]+' Prediction Performance', models_r2_mean, None)
i = i+1

j = 0
models_mae_scores = list()
models_rmse_scores = list()
models_r2_scores = list()
models_all_scores = list()
models_mae_scores_testing = list()
models_rmse_scores_testing = list()
models_mae_mean = list()
models_rmse_mean = list()
models_r2_mean = list()
models_mae_testing_mean = list()
models_rmse_testing_mean= list()

for model in get_models_BET():
    a,b,c,d,e,f =  repeated_Kfold(model, X, Y[:,i],0, LeaveOneOut(),splits,j)
    models_mae_scores.append(a)
    models_mae_mean.append(a.mean())
    models_rmse_scores.append(b)
    models_rmse_mean.append(b.mean())
    models_r2_scores.append(c)
    models_r2_mean.append(c.mean())
    models_all_scores.append(d)
    models_mae_scores_testing.append(e)
    models_mae_testing_mean.append(e.mean())
    models_rmse_scores_testing.append(f)
    models_rmse_testing_mean.append(f.mean())
    j=1+j

plot_result('Model Name', 'Mean Absolute Error', models_outputs[i]+' Prediction Performance', models_mae_mean, models_mae_testing_mean)
plot_result('Model Name', 'Root Mean Squared Error', models_outputs[i]+' Prediction Performancee', models_rmse_mean, models_rmse_testing_mean)
plot_result('Model Name', 'R\u00b2', models_outputs[i]+' Prediction Performance', models_r2_mean, None)
i = i+1


j = 0
models_mae_scores = list()
models_rmse_scores = list()
models_r2_scores = list()
models_all_scores = list()
models_mae_scores_testing = list()
models_rmse_scores_testing = list()
models_mae_mean = list()
models_rmse_mean = list()
models_r2_mean = list()
models_mae_testing_mean = list()
models_rmse_testing_mean= list()

for model in get_models_micro_volume():
    a,b,c,d,e,f =  repeated_Kfold(model, X, Y[:,i],0, LeaveOneOut(),splits,j)
    models_mae_scores.append(a)
    models_mae_mean.append(a.mean())
    models_rmse_scores.append(b)
    models_rmse_mean.append(b.mean())
    models_r2_scores.append(c)
    models_r2_mean.append(c.mean())
    models_all_scores.append(d)
    models_mae_scores_testing.append(e)
    models_mae_testing_mean.append(e.mean())
    models_rmse_scores_testing.append(f)
    models_rmse_testing_mean.append(f.mean())
    j=1+j

plot_result('Model Name', 'Mean Absolute Error', models_outputs[i]+' Prediction Performance', models_mae_mean, models_mae_testing_mean)
plot_result('Model Name', 'Root Mean Squared Error', models_outputs[i]+' Prediction Performancee', models_rmse_mean, models_rmse_testing_mean)
plot_result('Model Name', 'R\u00b2', models_outputs[i]+' Prediction Performance', models_r2_mean, None)
i = i+1


j = 0
models_mae_scores = list()
models_rmse_scores = list()
models_r2_scores = list()
models_all_scores = list()
models_mae_scores_testing = list()
models_rmse_scores_testing = list()
models_mae_mean = list()
models_rmse_mean = list()
models_r2_mean = list()
models_mae_testing_mean = list()
models_rmse_testing_mean= list()

for model in get_models_total_volume():
    a,b,c,d,e,f =  repeated_Kfold(model, X, Y[:,i],0, LeaveOneOut(),splits,j)
    models_mae_scores.append(a)
    models_mae_mean.append(a.mean())
    models_rmse_scores.append(b)
    models_rmse_mean.append(b.mean())
    models_r2_scores.append(c)
    models_r2_mean.append(c.mean())
    models_all_scores.append(d)
    models_mae_scores_testing.append(e)
    models_mae_testing_mean.append(e.mean())
    models_rmse_scores_testing.append(f)
    models_rmse_testing_mean.append(f.mean())
    j=1+j

plot_result('Model Name', 'Mean Absolute Error', models_outputs[i]+' Prediction Performance', models_mae_mean, models_mae_testing_mean)
plot_result('Model Name', 'Root Mean Squared Error', models_outputs[i]+' Prediction Performancee', models_rmse_mean, models_rmse_testing_mean)
plot_result('Model Name', 'R\u00b2', models_outputs[i]+' Prediction Performance', models_r2_mean, None)
i = i+1




