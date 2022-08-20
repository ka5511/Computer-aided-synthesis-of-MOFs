# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:28:39 2022

@author: Khalid Alotaibi
"""
# Purpose: Perform the indicated K-Fold cross-validations on 17 different algorithms to check MOF shape/quality prediction performance. Input to the random forest model consists of only the first 3 principal components (PC) obtained from the PCA analysis
####################################################################

# Step 1: Import needed libraries   
import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
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


# Models to be evaluated
from numpy import asarray
from numpy import polyfit
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import spearmanr

def get_models():
 	models = list()
 	models.append(LogisticRegression(random_state=0))
 	models.append(RidgeClassifier(random_state=0))
 	models.append(SGDClassifier(random_state=0))
 	models.append(PassiveAggressiveClassifier(random_state=0))
 	models.append(KNeighborsClassifier())
 	models.append(DecisionTreeClassifier(random_state=0))
 	models.append(ExtraTreeClassifier(random_state=0))
 	models.append(LinearSVC(random_state=0))
 	models.append(SVC(random_state=0))
 	models.append(GaussianNB())
 	models.append(AdaBoostClassifier(random_state=0))
 	models.append(BaggingClassifier(random_state=0))
 	models.append(RandomForestClassifier(criterion="entropy",min_samples_split=2,random_state=0))
 	models.append(ExtraTreesClassifier(random_state=0))
 	models.append(GaussianProcessClassifier(random_state=0))
 	models.append(GradientBoostingClassifier(random_state=0))
 	return models

models_list = ['LR','Ridge','SGDC','PAC','KNC','DTC','ETC','LSVC','SVC','GNB','ABC','BC','RF','ETsC','GPC','GBC']
models_list_expanded = ['LogisticRegression','RidgeClassifier','SGDClassifier','PassiveAggressiveClassifier','KNeighborsClassifier','DecisionTreeClassifier','ExtraTreeClassifier','LinearSVC','SVC','GaussianNB','AdaBoostClassifier','BaggingClassifier','RandomForestClassifier','ExtraTreesClassifier','GaussianProcessClassifier','GradientBoostingClassifier']

# Use all models for cross validation for MOF Shape
i = 0
highest_mcc = list()
highest_mcc_score = list()
for model in get_models():
    ideal_values = cross_validation(model,principal_components,encoded_y2,LeaveOneOut())
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
        cv = KFold(n_splits=k, shuffle=True, random_state=1)
        results = cross_validation(model, principal_components, encoded_y2, cv)
        # Evaluate k values
        k_mean = results["Mean Validation Accuracy"]
        k_min = results["Validation Accuracy scores"].min()
        k_max = results["Validation Accuracy scores"].max()
        # Report performance
        a_means.append(k_mean)
        a_mins.append(k_mean-k_min)
        a_maxs.append(k_max-k_mean)
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
    plt.title("MOF Shape Accuracy Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, f_means, yerr=[f_mins, f_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [f_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("MOF Shape F1 Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, mat_means, yerr=[mat_mins, mat_maxs], fmt='o')
    # show the plot
    plt.title("MOF Shape MCC Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("MCC Score", fontsize=14)
    plt.grid()
    plt.show()
    highest_mcc.append(mat_means.index(max(mat_means))+2)
    highest_mcc_score.append(max(mat_means))
    i = i+1
    

plt.bar(models_list,highest_mcc)
plt.title("KFold with Highest Mean MCC Score for MOF Shape for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("KFold Number", fontsize=14)
plt.show()

RF_mcc_score = [highest_mcc_score[12]]*len(models_list)
plt.bar(models_list,highest_mcc_score)
plt.plot(models_list, RF_mcc_score, color='r',label="Random Forest Score")
plt.title("Highest Mean MCC Score for MOF Shape for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("MCC Score", fontsize=14)
plt.show()


# Use all models for cross validation for MOF Quality
i = 0
highest_mcc = list()
highest_mcc_score = list()
for model in get_models():
    ideal_values = cross_validation(model,principal_components,encoded_y3,LeaveOneOut())
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
        cv = KFold(n_splits=k, shuffle=True, random_state=1)
        results = cross_validation(model, principal_components, encoded_y3, cv)
        # Evaluate k values
        k_mean = results["Mean Validation Accuracy"]
        k_min = results["Validation Accuracy scores"].min()
        k_max = results["Validation Accuracy scores"].max()
        # Report performance
        a_means.append(k_mean)
        a_mins.append(k_mean-k_min)
        a_maxs.append(k_max-k_mean)
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
    plt.title("MOF Quality Accuracy Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, f_means, yerr=[f_mins, f_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [f_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("MOF Quality F1 Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, mat_means, yerr=[mat_mins, mat_maxs], fmt='o')
    # show the plot
    plt.title("MOF Quality MCC Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("MCC Score", fontsize=14)
    plt.grid()
    plt.show()
    highest_mcc.append(mat_means.index(max(mat_means))+2)
    highest_mcc_score.append(max(mat_means))
    i = i+1
    

plt.bar(models_list,highest_mcc)
plt.title("KFold with Highest Mean MCC Score for MOF Quality for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("KFold Number", fontsize=14)
plt.show()

RF_mcc_score = [highest_mcc_score[12]]*len(models_list)
plt.bar(models_list,highest_mcc_score)
plt.plot(models_list, RF_mcc_score, color='r',label="Random Forest Score")
plt.title("Highest Mean MCC Score for MOF Quality for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("MCC Score", fontsize=14)
plt.show()

# #############################################################################
# #                           Repeated KFold Section                          #
# #############################################################################

# Use all models for cross validation for MOF Shape
i = 0
highest_mcc = list()
highest_mcc_score = list()
for model in get_models():
    ideal_values = cross_validation(model,principal_components,encoded_y2,LeaveOneOut())
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
        cv = RepeatedKFold(n_splits=k, n_repeats=10, random_state=1)
        results = cross_validation(model, principal_components, encoded_y2, cv)
        # Evaluate k values
        k_mean = results["Mean Validation Accuracy"]
        k_min = results["Validation Accuracy scores"].min()
        k_max = results["Validation Accuracy scores"].max()
        # Report performance
        a_means.append(k_mean)
        a_mins.append(k_mean-k_min)
        a_maxs.append(k_max-k_mean)
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
    plt.title("MOF Shape Accuracy Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, f_means, yerr=[f_mins, f_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [f_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("MOF Shape F1 Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, mat_means, yerr=[mat_mins, mat_maxs], fmt='o')
    # show the plot
    plt.title("MOF Shape MCC Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("MCC Score", fontsize=14)
    plt.grid()
    plt.show()
    highest_mcc.append(mat_means.index(max(mat_means))+2)
    highest_mcc_score.append(max(mat_means))
    i = i+1
    

plt.bar(models_list,highest_mcc)
plt.title("Repeated KFold with Highest Mean MCC Score for MOF Shape for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("Repeated KFold Number", fontsize=14)
plt.show()

RF_mcc_score = [highest_mcc_score[12]]*len(models_list)
plt.bar(models_list,highest_mcc_score)
plt.plot(models_list, RF_mcc_score, color='r',label="Random Forest Score")
plt.title("Highest Mean MCC Score for MOF Shape for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("MCC Score", fontsize=14)
plt.show()


# Use all models for cross validation for MOF Quality
i = 0
highest_mcc = list()
highest_mcc_score = list()
for model in get_models():
    ideal_values = cross_validation(model,principal_components,encoded_y3,LeaveOneOut())
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
        cv = RepeatedKFold(n_splits=k, n_repeats=10, random_state=1)
        results = cross_validation(model, principal_components, encoded_y3, cv)
        # Evaluate k values
        k_mean = results["Mean Validation Accuracy"]
        k_min = results["Validation Accuracy scores"].min()
        k_max = results["Validation Accuracy scores"].max()
        # Report performance
        a_means.append(k_mean)
        a_mins.append(k_mean-k_min)
        a_maxs.append(k_max-k_mean)
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
    plt.title("MOF Quality Accuracy Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, f_means, yerr=[f_mins, f_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [f_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("MOF Quality F1 Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, mat_means, yerr=[mat_mins, mat_maxs], fmt='o')
    # show the plot
    plt.title("MOF Quality MCC Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("MCC Score", fontsize=14)
    plt.grid()
    plt.show()
    highest_mcc.append(mat_means.index(max(mat_means))+2)
    highest_mcc_score.append(max(mat_means))
    i = i+1
    

plt.bar(models_list,highest_mcc)
plt.title("Repeated KFold with Highest Mean MCC Score for MOF Quality for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("Repeated KFold Number", fontsize=14)
plt.show()

RF_mcc_score = [highest_mcc_score[12]]*len(models_list)
plt.bar(models_list,highest_mcc_score)
plt.plot(models_list, RF_mcc_score, color='r',label="Random Forest Score")
plt.title("Highest Mean MCC Score for MOF Quality for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("MCC Score", fontsize=14)
plt.show()


#############################################################################
#                           Repeated Stratified KFold Section               #
#############################################################################

# Use all models for cross validation for MOF Shape
i = 0
highest_mcc = list()
highest_mcc_score = list()
for model in get_models():
    ideal_values = cross_validation(model,principal_components,encoded_y2,LeaveOneOut())
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
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=1)
        results = cross_validation(model, principal_components, encoded_y2, cv)
        # Evaluate k values
        k_mean = results["Mean Validation Accuracy"]
        k_min = results["Validation Accuracy scores"].min()
        k_max = results["Validation Accuracy scores"].max()
        # Report performance
        a_means.append(k_mean)
        a_mins.append(k_mean-k_min)
        a_maxs.append(k_max-k_mean)
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
    plt.title("MOF Shape Accuracy Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, f_means, yerr=[f_mins, f_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [f_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("MOF Shape F1 Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, mat_means, yerr=[mat_mins, mat_maxs], fmt='o')
    # show the plot
    plt.title("MOF Shape MCC Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("MCC Score", fontsize=14)
    plt.grid()
    plt.show()
    highest_mcc.append(mat_means.index(max(mat_means))+2)
    highest_mcc_score.append(max(mat_means))
    i = i+1
    

plt.bar(models_list,highest_mcc)
plt.title("Repeated Stratified KFold with Highest Mean MCC Score for MOF Shape for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("Repeated KFold Number", fontsize=14)
plt.show()

RF_mcc_score = [highest_mcc_score[12]]*len(models_list)
plt.bar(models_list,highest_mcc_score)
plt.plot(models_list, RF_mcc_score, color='r',label="Random Forest Score")
plt.title("Highest Mean MCC Score for MOF Shape for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("MCC Score", fontsize=14)
plt.show()


# Use all models for cross validation for MOF Quality
i = 0
highest_mcc = list()
highest_mcc_score = list()
for model in get_models():
    ideal_values = cross_validation(model,principal_components,encoded_y3,LeaveOneOut())
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
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=10, random_state=1)
        results = cross_validation(model, principal_components, encoded_y3, cv)
        # Evaluate k values
        k_mean = results["Mean Validation Accuracy"]
        k_min = results["Validation Accuracy scores"].min()
        k_max = results["Validation Accuracy scores"].max()
        # Report performance
        a_means.append(k_mean)
        a_mins.append(k_mean-k_min)
        a_maxs.append(k_max-k_mean)
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
    plt.title("MOF Quality Accuracy Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("Accuracy Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, f_means, yerr=[f_mins, f_maxs], fmt='o')
    # plot the ideal case in a separate color
    plt.plot(folds, [f_ideal for _ in range(len(folds))], color='r',label="LOOCV Score")
    # show the plot
    plt.title("MOF Quality F1 Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()
    # line plot of k mean values with min/max error bars
    plt.errorbar(folds, mat_means, yerr=[mat_mins, mat_maxs], fmt='o')
    # show the plot
    plt.title("MOF Quality MCC Scores for Different K Values - "+str(models_list_expanded[i]), fontsize=15)
    plt.xlabel("K Value", fontsize=14)
    plt.ylabel("MCC Score", fontsize=14)
    plt.grid()
    plt.show()
    highest_mcc.append(mat_means.index(max(mat_means))+2)
    highest_mcc_score.append(max(mat_means))
    i = i+1
    

plt.bar(models_list,highest_mcc)
plt.title("Repeated Stratified KFold with Highest Mean MCC Score for MOF Quality for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("Repeated KFold Number", fontsize=14)
plt.show()

RF_mcc_score = [highest_mcc_score[12]]*len(models_list)
plt.bar(models_list,highest_mcc_score)
plt.plot(models_list, RF_mcc_score, color='r',label="Random Forest Score")
plt.title("Highest Mean MCC Score for MOF Quality for Different Algorithms", fontsize=15)
plt.xlabel("Algorithm", fontsize=14)
plt.xticks(size = 7)
plt.ylabel("MCC Score", fontsize=14)
plt.show()
