# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:54:05 2022

@author: Khalid Alotaibi
"""
# Purpose: Perform bootstrap sampling to train selected regression algorithms (either base or optimized version) and check their prediction performance. Input to these  models consists of only the first 3 principal components (PC) obtained from the PCA analysis. If data augmentation is applied, the applied technique is reflected on the file title.
################################################
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
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import resample
import joblib
from statistics import mean
from statistics import median

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
data_density = np.column_stack((X,Y[:,0]))
data_size = np.column_stack((X,Y[:,1]))
data_BET = np.column_stack((X,Y[:,2]))
data_micro_volume = np.column_stack((X,Y[:,3]))
data_total_volume = np.column_stack((X,Y[:,4]))



def get_models_density():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(SVR())
    return models

def get_models_size():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(SVR())
    return models

def get_models_BET():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(SVR())
    return models

def get_models_micro_volume():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(SVR())
    return models


def get_models_total_volume():
    models = list()
    models.append(RandomForestRegressor(n_estimators=100,random_state=0))
    models.append(LinearRegression())
    models.append(Ridge(random_state=0))
    models.append(linear_model.Lasso(random_state=0))
    models.append(KNeighborsRegressor(n_neighbors=4))
    models.append(SVR())
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

# configure bootstrap with the number of sample sets and the size of each sample
n_iterations = 10000
n_size = int(len(data_density) * 0.8)
# Variable to identify the model name from models_list
j=0
# Variable to identify output name from the models_outputs
k = 0
# run bootstrap for unoptimized model
for model in get_models_density():
    stats_mae_training = list()
    stats_mae_testing = list()
    stats_rmse_training = list()
    stats_rmse_testing = list()
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(data_density, n_samples=n_size)
        test = np.array([x for x in data_density if x.tolist() not in train.tolist()])
        # fit model
        model.fit(train[:,:-1], train[:,-1])
        # evaluate model
        predictions_train = model.predict(train[:,:-1])
        mae_train = np.mean(np.abs(train[:,-1]-predictions_train))
        stats_mae_training.append(mae_train)
        rmse_train = np.sqrt(np.mean((train[:,-1]-predictions_train)**2))
        stats_rmse_training.append(rmse_train)
        
        
        predictions_test = model.predict(test[:,:-1])    
        mae_test = np.mean(np.abs(test[:,-1]-predictions_test))
        stats_mae_testing.append(mae_test)
        rmse_test = np.sqrt(np.mean((test[:,-1]-predictions_test)**2))
        stats_rmse_testing.append(rmse_test)
        print("Completion percentage: ", i/n_iterations*100,"%")
    
    # plot mae training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_training)))
    plt.vlines(median(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_testing)))
    plt.vlines(median(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))    
    
    
    # plot rmse training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_training)))
    plt.vlines(median(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_testing)))
    plt.vlines(median(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))          
    j = j+1
    
k = k+1
j=0

# run bootstrap for unoptimized model
for model in get_models_size():
    stats_mae_training = list()
    stats_mae_testing = list()
    stats_rmse_training = list()
    stats_rmse_testing = list()
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(data_size, n_samples=n_size)
        test = np.array([x for x in data_size if x.tolist() not in train.tolist()])
        # fit model
        model.fit(train[:,:-1], train[:,-1])
        # evaluate model
        predictions_train = model.predict(train[:,:-1])
        mae_train = np.mean(np.abs(train[:,-1]-predictions_train))
        stats_mae_training.append(mae_train)
        rmse_train = np.sqrt(np.mean((train[:,-1]-predictions_train)**2))
        stats_rmse_training.append(rmse_train)
        
        
        predictions_test = model.predict(test[:,:-1])    
        mae_test = np.mean(np.abs(test[:,-1]-predictions_test))
        stats_mae_testing.append(mae_test)
        rmse_test = np.sqrt(np.mean((test[:,-1]-predictions_test)**2))
        stats_rmse_testing.append(rmse_test)
        print("Completion percentage: ", i/n_iterations*100,"%")
    
    # plot mae training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_training)))
    plt.vlines(median(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_testing)))
    plt.vlines(median(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))    
    
    
    # plot rmse training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_training)))
    plt.vlines(median(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_testing)))
    plt.vlines(median(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))          
    j = j+1
    
k = k+1
j=0


# run bootstrap for unoptimized model
for model in get_models_BET():
    stats_mae_training = list()
    stats_mae_testing = list()
    stats_rmse_training = list()
    stats_rmse_testing = list()
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(data_BET, n_samples=n_size)
        test = np.array([x for x in data_BET if x.tolist() not in train.tolist()])
        # fit model
        model.fit(train[:,:-1], train[:,-1])
        # evaluate model
        predictions_train = model.predict(train[:,:-1])
        mae_train = np.mean(np.abs(train[:,-1]-predictions_train))
        stats_mae_training.append(mae_train)
        rmse_train = np.sqrt(np.mean((train[:,-1]-predictions_train)**2))
        stats_rmse_training.append(rmse_train)
        
        
        predictions_test = model.predict(test[:,:-1])    
        mae_test = np.mean(np.abs(test[:,-1]-predictions_test))
        stats_mae_testing.append(mae_test)
        rmse_test = np.sqrt(np.mean((test[:,-1]-predictions_test)**2))
        stats_rmse_testing.append(rmse_test)
        print("Completion percentage: ", i/n_iterations*100,"%")
    
    # plot mae training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_training)))
    plt.vlines(median(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_testing)))
    plt.vlines(median(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))    
    
    
    # plot rmse training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_training)))
    plt.vlines(median(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_testing)))
    plt.vlines(median(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))          
    j = j+1
    
k = k+1
j=0

# run bootstrap for unoptimized model
for model in get_models_micro_volume():
    stats_mae_training = list()
    stats_mae_testing = list()
    stats_rmse_training = list()
    stats_rmse_testing = list()
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(data_micro_volume, n_samples=n_size)
        test = np.array([x for x in data_micro_volume if x.tolist() not in train.tolist()])
        # fit model
        model.fit(train[:,:-1], train[:,-1])
        # evaluate model
        predictions_train = model.predict(train[:,:-1])
        mae_train = np.mean(np.abs(train[:,-1]-predictions_train))
        stats_mae_training.append(mae_train)
        rmse_train = np.sqrt(np.mean((train[:,-1]-predictions_train)**2))
        stats_rmse_training.append(rmse_train)
        
        
        predictions_test = model.predict(test[:,:-1])    
        mae_test = np.mean(np.abs(test[:,-1]-predictions_test))
        stats_mae_testing.append(mae_test)
        rmse_test = np.sqrt(np.mean((test[:,-1]-predictions_test)**2))
        stats_rmse_testing.append(rmse_test)
        print("Completion percentage: ", i/n_iterations*100,"%")
    
    # plot mae training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_training)))
    plt.vlines(median(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_testing)))
    plt.vlines(median(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))    
    
    
    # plot rmse training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_training)))
    plt.vlines(median(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_testing)))
    plt.vlines(median(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))          
    j = j+1
    
k = k+1
j=0

# run bootstrap for unoptimized model
for model in get_models_total_volume():
    stats_mae_training = list()
    stats_mae_testing = list()
    stats_rmse_training = list()
    stats_rmse_testing = list()
    for i in range(n_iterations):
        # prepare train and test sets
        train = resample(data_total_volume, n_samples=n_size)
        test = np.array([x for x in data_total_volume if x.tolist() not in train.tolist()])
        # fit model
        model.fit(train[:,:-1], train[:,-1])
        # evaluate model
        predictions_train = model.predict(train[:,:-1])
        mae_train = np.mean(np.abs(train[:,-1]-predictions_train))
        stats_mae_training.append(mae_train)
        rmse_train = np.sqrt(np.mean((train[:,-1]-predictions_train)**2))
        stats_rmse_training.append(rmse_train)
        
        
        predictions_test = model.predict(test[:,:-1])    
        mae_test = np.mean(np.abs(test[:,-1]-predictions_test))
        stats_mae_testing.append(mae_test)
        rmse_test = np.sqrt(np.mean((test[:,-1]-predictions_test)**2))
        stats_rmse_testing.append(rmse_test)
        print("Completion percentage: ", i/n_iterations*100,"%")
    
    # plot mae training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_training)))
    plt.vlines(median(stats_mae_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mae_testing)))
    plt.vlines(median(stats_mae_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mae_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_mae_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('MAE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_mae_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_mae_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))    
    
    
    # plot rmse training scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_training)))
    plt.vlines(median(stats_rmse_training), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_training)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_training, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Training')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_training, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_training, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    # plot mae testing scores
    plt.title(models_outputs[k]+" for "+models_list_expanded[j])
    plt.vlines(mean(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_rmse_testing)))
    plt.vlines(median(stats_rmse_testing), [0], 2000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_rmse_testing)),color="C1")
    alpha = 0.9
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
    plt.hist(stats_rmse_testing, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
    plt.xlabel('RMSE for Testing')
    plt.ylabel('Count')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    # confidence intervals
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats_rmse_testing, p))
    asl = lower
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1000, np.percentile(stats_rmse_testing, p))
    asu = upper
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))          
    j = j+1
    
