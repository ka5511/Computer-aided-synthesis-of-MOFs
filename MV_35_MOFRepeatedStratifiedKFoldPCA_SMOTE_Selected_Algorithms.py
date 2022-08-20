# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:28:39 2022

@author: khalid alotaibi
"""
# Purpose: Perform the indicated K-Fold cross-validations on selected algorithms to check MOF shape/quality prediction performance. Input to these  models consists of only the first 3 principal components (PC) obtained from the PCA analysis. If data augmentation is applied, the applied technique is reflected on the file title.
####################################################################

# Step 1: Import needed libraries   
import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import std
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from numpy import std
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
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
            labels.append(str(i+1))
        X_axis = np.arange(len(labels))
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
            labels.append(str(i+1))
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

def get_models_shape():
 	models = list()
 	models.append(LogisticRegression(random_state=0))
 	models.append(RidgeClassifier(random_state=0))
 	models.append(GaussianProcessClassifier(random_state=0))
 	return models

def get_models_quality():
 	models = list()
 	models.append(DecisionTreeClassifier(random_state=0))
 	models.append(LinearSVC(random_state=0))
 	models.append(GaussianNB())
 	return models

# performing preprocessing part to scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)

principal_components = pca.fit_transform(X_scaled)


models_list = ['LR','Ridge','GPC','DTC','LSVC','GNB']
models_list_expanded = ['SMOTE Logistic Regression','SMOTE Ridge Classifier','SMOTE Gaussian Process Classifier','SMOTE Decision Tree Classifier','SMOTE Linear SVC','SMOTE Gaussian NB']
models_results_shape = list()
splits = 3
i=0
for model in get_models_shape():
    # Define a pipeline
    steps = [('over', BorderlineSMOTE(k_neighbors=1)), ('model', model)]
    pipeline = Pipeline(steps=steps)
    # Use pipeline
    models_results_shape.append(cross_validation(pipeline, principal_components, encoded_y2, RepeatedStratifiedKFold(n_splits = splits,random_state=1)))
    plot_result(models_list_expanded[i]+"Repeated Stratified K-Fold", "Accuracy", "MOF Shape Accuracy Scores in "+str(splits)+" Folds", models_results_shape[i]["Training Accuracy scores"], models_results_shape[i]["Validation Accuracy scores"], models_results_shape[i]["Mean Validation Accuracy"])
    plot_result(models_list_expanded[i]+"Repeated Stratified K-Fold", "Precision", "MOF Shape Precision Scores in "+str(splits)+" Folds", models_results_shape[i]["Training Precision scores"], models_results_shape[i]["Validation Precision scores"], models_results_shape[i]["Mean Validation Precision"])
    plot_result(models_list_expanded[i]+"Repeated Stratified K-Fold", "Recall", "MOF Shape Recall Scores in "+str(splits)+" Folds", models_results_shape[i]["Training Recall scores"], models_results_shape[i]["Validation Recall scores"], models_results_shape[i]["Mean Validation Recall"])
    plot_result(models_list_expanded[i]+"Repeated Stratified K-Fold", "F1", "MOF Shape F1 Scores in "+str(splits)+" Folds", models_results_shape[i]["Training F1 scores"], models_results_shape[i]["Validation F1 scores"], models_results_shape[i]["Mean Validation F1 Score"])
    plot_result_MCC(models_list_expanded[i]+"Repeated Stratified K-Fold", "MCC", "MOF Shape MCC Scores in "+str(splits)+" Folds", models_results_shape[i]["MCC"], models_results_shape[i]["MCC"].mean())
    ideal_values = cross_validation(pipeline,principal_components,encoded_y2,LeaveOneOut())
    a_ideal = ideal_values["Mean Validation Accuracy"]
    p_ideal = ideal_values["Mean Validation Precision"]
    r_ideal = ideal_values["Mean Validation Recall"]
    f_ideal = ideal_values["Mean Validation F1 Score"]
    mcc_ideal = ideal_values["MCC"].mean()
    # Define the number of folds to test
    folds = range(2,11)
    a_means, a_mins, a_maxs = list(),list(),list()
    p_means, p_mins, p_maxs = list(),list(),list()
    r_means, r_mins, r_maxs = list(),list(),list()
    f_means, f_mins, f_maxs = list(),list(),list()
    mat_means, mat_mins, mat_maxs = list(),list(),list()
    # evaluate each k value
    for k in folds:
        # Define test conditions
        cv = RepeatedStratifiedKFold(n_splits=k, random_state=1)
        results = cross_validation(pipeline, principal_components, encoded_y2, cv)
        # Evaluate k values
        k_mean = results["Mean Validation Accuracy"]
        k_min = results["Validation Accuracy scores"].min()
        k_max = results["Validation Accuracy scores"].max()
        # Report performance
        a_means.append(k_mean)
        a_mins.append(k_mean-k_min)
        a_maxs.append(k_max-k_mean)
        # Evaluate k values
        k_mean = results["Mean Validation Precision"]
        k_min = results["Validation Precision scores"].min()
        k_max = results["Validation Precision scores"].max()
        # Report performance
        p_means.append(k_mean)
        p_mins.append(k_mean-k_min)
        p_maxs.append(k_max-k_mean)
        # Evaluate k values
        k_mean = results["Mean Validation Recall"]
        k_min = results["Validation Recall scores"].min()
        k_max = results["Validation Recall scores"].max()
        # Report performance
        r_means.append(k_mean)
        r_mins.append(k_mean-k_min)
        r_maxs.append(k_max-k_mean)
        # Evaluate k values
        k_mean = results["Mean Validation F1 Score"]
        k_min = results["Validation F1 scores"].min()
        k_max = results["Validation F1 scores"].max()
        # Report performance
        f_means.append(k_mean)
        f_mins.append(k_mean-k_min)
        f_maxs.append(k_max-k_mean)
        # Evaluate k values
        k_mean = results["MCC"].mean()
        k_min = results["MCC"].min()
        k_max = results["MCC"].max()
        # Report performance
        mat_means.append(k_mean)
        mat_mins.append(k_mean-k_min)
        mat_maxs.append(k_max-k_mean)

    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, a_means, yerr=[a_mins, a_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [a_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("Repeated Stratified KFold MOF Shape Accuracy Scores for Different K Values"+models_list_expanded[i], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, p_means, yerr=[p_mins, p_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [p_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("Repeated Stratified KFold MOF Shape Precision Scores for Different K Values"+models_list_expanded[i], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Precision Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, r_means, yerr=[r_mins, r_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [r_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("Repeated Stratified KFold MOF Shape Recall Scores for Different K Values"+models_list_expanded[i], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Recall Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, f_means, yerr=[f_mins, f_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [f_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("Repeated Stratified KFold MOF Shape F1 Scores for Different K Values"+models_list_expanded[i], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, mat_means, yerr=[mat_mins, mat_maxs], fmt='o')
    # show the plot
    plt.title("Repeated Stratified KFold MOF Shape MCC Scores for Different K Values"+models_list_expanded[i], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("MCC Score", fontsize=14)
    plt.grid()
    plt.show()
    i=i+1
 

##############################################################################
#                               Quality Analysis                             #
##############################################################################
models_results_quality = list()
i=0
for model in get_models_quality():
    # Define a pipeline
    steps = [('over', BorderlineSMOTE(k_neighbors=1)), ('model', model)]
    pipeline = Pipeline(steps=steps)
    # Use pipeline
    models_results_quality.append(cross_validation(pipeline, principal_components, encoded_y3, RepeatedStratifiedKFold(n_splits=splits, random_state=1)))
    plot_result(models_list_expanded[i+3]+"Repeated Stratified KFold", "Accuracy", "Repeated KFold MOF Quality Accuracy Scores in "+str(splits)+" Folds", models_results_quality[i]["Training Accuracy scores"], models_results_quality[i]["Validation Accuracy scores"], models_results_quality[i]["Mean Validation Accuracy"])
    plot_result(models_list_expanded[i+3]+"Repeated Stratified KFold", "Precision", "Repeated KFold MOF Quality Precision Scores in "+str(splits)+" Folds", models_results_quality[i]["Training Precision scores"], models_results_quality[i]["Validation Precision scores"], models_results_quality[i]["Mean Validation Precision"])
    plot_result(models_list_expanded[i+3]+"Repeated Stratified KFold", "Recall", "Repeated KFold MOF Quality Recall Scores in "+str(splits)+" Folds", models_results_quality[i]["Training Recall scores"], models_results_quality[i]["Validation Recall scores"], models_results_quality[i]["Mean Validation Recall"])
    plot_result(models_list_expanded[i+3]+"Repeated Stratified KFold", "F1", "Repeated KFold MOF Quality F1 Scores in "+str(splits)+" Folds", models_results_quality[i]["Training F1 scores"], models_results_quality[i]["Validation F1 scores"], models_results_quality[i]["Mean Validation F1 Score"])
    plot_result_MCC(models_list_expanded[i+3]+"Repeated KFold", "MCC", "Repeated KFold MOF Quality MCC Scores in "+str(splits)+" Folds", models_results_quality[i]["MCC"], models_results_quality[i]["MCC"].mean())
    ideal_values = cross_validation(pipeline,principal_components,encoded_y3,LeaveOneOut())
    a_ideal = ideal_values["Mean Validation Accuracy"]
    p_ideal = ideal_values["Mean Validation Precision"]
    r_ideal = ideal_values["Mean Validation Recall"]
    f_ideal = ideal_values["Mean Validation F1 Score"]
    mcc_ideal = ideal_values["MCC"].mean()
    # Define the number of folds to test
    folds = range(2,11)
    a_means, a_mins, a_maxs = list(),list(),list()
    p_means, p_mins, p_maxs = list(),list(),list()
    r_means, r_mins, r_maxs = list(),list(),list()
    f_means, f_mins, f_maxs = list(),list(),list()
    mat_means, mat_mins, mat_maxs = list(),list(),list()
    # evaluate each k value
    for k in folds:
        # Define test conditions
        cv = RepeatedStratifiedKFold(n_splits=k, random_state=1)
        results = cross_validation(pipeline, principal_components, encoded_y3, cv)
        # Evaluate k values
        k_mean = results["Mean Validation Accuracy"]
        k_min = results["Validation Accuracy scores"].min()
        k_max = results["Validation Accuracy scores"].max()
        # Report performance
        a_means.append(k_mean)
        a_mins.append(k_mean-k_min)
        a_maxs.append(k_max-k_mean)
        # Evaluate k values
        k_mean = results["Mean Validation Precision"]
        k_min = results["Validation Precision scores"].min()
        k_max = results["Validation Precision scores"].max()
        # Report performance
        p_means.append(k_mean)
        p_mins.append(k_mean-k_min)
        p_maxs.append(k_max-k_mean)
        # Evaluate k values
        k_mean = results["Mean Validation Recall"]
        k_min = results["Validation Recall scores"].min()
        k_max = results["Validation Recall scores"].max()
        # Report performance
        r_means.append(k_mean)
        r_mins.append(k_mean-k_min)
        r_maxs.append(k_max-k_mean)
        # Evaluate k values
        k_mean = results["Mean Validation F1 Score"]
        k_min = results["Validation F1 scores"].min()
        k_max = results["Validation F1 scores"].max()
        # Report performance
        f_means.append(k_mean)
        f_mins.append(k_mean-k_min)
        f_maxs.append(k_max-k_mean)
        # Evaluate k values
        k_mean = results["MCC"].mean()
        k_min = results["MCC"].min()
        k_max = results["MCC"].max()
        # Report performance
        mat_means.append(k_mean)
        mat_mins.append(k_mean-k_min)
        mat_maxs.append(k_max-k_mean)

    
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, a_means, yerr=[a_mins, a_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [a_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("Repeated Stratified KFold MOF Quality Accuracy Scores for Different K Values"+models_list_expanded[i+3], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, p_means, yerr=[p_mins, p_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [p_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("Repeated Stratified KFold MOF Quality Precision Scores for Different K Values"+models_list_expanded[i+3], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Precision Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, r_means, yerr=[r_mins, r_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [r_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("Repeated Stratified KFold MOF Quality Recall Scores for Different K Values"+models_list_expanded[i+3], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Recall Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, f_means, yerr=[f_mins, f_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [f_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("Repeated Stratified KFold MOF Quality F1 Scores for Different K Values"+models_list_expanded[i+3], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, mat_means, yerr=[mat_mins, mat_maxs], fmt='o')
    # show the plot
    plt.title("Repeated Stratified KFold MOF Quality MCC Scores for Different K Values"+models_list_expanded[i+3], fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("MCC Score", fontsize=14)
    plt.grid()
    plt.show()
    i=i+1
     
