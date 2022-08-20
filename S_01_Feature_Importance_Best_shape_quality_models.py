# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:28:39 2022

@author: Khalid Alotaibi
"""
# Purpose: Obtain feature importance weight for each feature in the dataset on the prediction of MOF shape/quality using random forest models by averaging weight scores over mulitple bootsampling sets
####################################################################

# Step 1: Import needed libraries   
import numpy as np
import pandas
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
from numpy import std
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib
from statistics import mean
from statistics import median
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

# performing preprocessing part to scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)


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

data_shape = np.column_stack((X_scaled,encoded_y2))
data_quality = np.column_stack((X_scaled,encoded_y3))

# Import the optimized models
filename2 = "RF_Randomized_Optimized_Shape.joblib"
rf_random2 = joblib.load(filename2)

rf_random3 = RandomForestClassifier(random_state=0)

rf_all_features = rf_random2.best_estimator_.fit(X_scaled,encoded_y2)

# configure bootstrap with the number of sample sets and the size of each sample
n_iterations = 1000
n_size = int(len(data_shape) * 1)
# run bootstrap for unoptimized model
stats_accuracy_shape = list()
stats_mcc_shape = list()
feature_importance_element_0= list()
feature_importance_element_1= list()
feature_importance_element_2= list()
feature_importance_element_3= list()
feature_importance_element_4= list()
feature_importance_element_5= list()
feature_importance_element_6= list()

for i in range(n_iterations):
    # prepare train and test sets
    train = resample(data_shape, n_samples=n_size, stratify=data_shape[:,-1])
    test = np.array([x for x in data_shape if x.tolist() not in train.tolist()])
    # fit model
    model = rf_random2.best_estimator_.fit(train[:,:-1], train[:,-1])
    feature_importance_element_0.append(model.feature_importances_[0])
    feature_importance_element_1.append(model.feature_importances_[1])
    feature_importance_element_2.append(model.feature_importances_[2])
    feature_importance_element_3.append(model.feature_importances_[3])
    feature_importance_element_4.append(model.feature_importances_[4])
    feature_importance_element_5.append(model.feature_importances_[5])
    feature_importance_element_6.append(model.feature_importances_[6])
    # evaluate model
    predictions = model.predict(test[:,:-1])
    score_accuracy = accuracy_score(test[:,-1], predictions)
    print("Accuracy: ", score_accuracy)
    stats_accuracy_shape.append(score_accuracy)
    score_mcc = matthews_corrcoef(test[:,-1], predictions)
    print("MCC: ", score_mcc)
    print("Completion percentage: ", i/n_iterations*100,"%")
    stats_mcc_shape.append(score_mcc)

average_feature_importance_element = list()
average_feature_importance_element.append(mean(feature_importance_element_0))
average_feature_importance_element.append(mean(feature_importance_element_1))
average_feature_importance_element.append(mean(feature_importance_element_2))
average_feature_importance_element.append(mean(feature_importance_element_3))
average_feature_importance_element.append(mean(feature_importance_element_4))
average_feature_importance_element.append(mean(feature_importance_element_5))
average_feature_importance_element.append(mean(feature_importance_element_6))

# plot accuracy scores
plt.title("Accuracy Shape for RF")
plt.vlines(mean(stats_accuracy_shape), [0], 3000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_accuracy_shape)))
plt.vlines(median(stats_accuracy_shape), [0], 3000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_accuracy_shape)),color="C1")
alpha = 0.9
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats_accuracy_shape, p))
plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats_accuracy_shape, p))
plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
plt.hist(stats_accuracy_shape, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.legend(loc="upper left")
plt.grid()
plt.show()
# confidence intervals
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats_accuracy_shape, p))
asl = lower
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats_accuracy_shape, p))
asu = upper
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    
# plot MCC scores
plt.title("MCC Shape for RF")
plt.vlines(mean(stats_mcc_shape), [0], 3000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mcc_shape)))
plt.vlines(median(stats_mcc_shape), [0], 3000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mcc_shape)),color="C1")
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats_mcc_shape, p))
plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats_mcc_shape, p))
plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
plt.hist(stats_mcc_shape, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
plt.xlabel('MCC')
plt.ylabel('Count')
plt.legend(loc="upper left")
plt.grid()
plt.show()
# confidence intervals
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats_mcc_shape, p))
msl = lower
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats_mcc_shape, p))
msu = upper
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))





quality_feature_importance_element_0= list()
quality_feature_importance_element_1= list()
quality_feature_importance_element_2= list()
quality_feature_importance_element_3= list()
quality_feature_importance_element_4= list()
quality_feature_importance_element_5= list()
quality_feature_importance_element_6= list()

# run bootstrap for unoptimized model
MCC_quality_scores = list()
stats_accuracy_quality = list()
stats_mcc_quality = list()
for i in range(n_iterations):
    # prepare train and test sets
    train = resample(data_quality, n_samples=n_size, stratify=data_quality[:,-1])
    test = np.array([x for x in data_quality if x.tolist() not in train.tolist()])
    # fit model
    model = rf_random3.fit(train[:,:-1], train[:,-1])
    quality_feature_importance_element_0.append(model.feature_importances_[0])
    quality_feature_importance_element_1.append(model.feature_importances_[1])
    quality_feature_importance_element_2.append(model.feature_importances_[2])
    quality_feature_importance_element_3.append(model.feature_importances_[3])
    quality_feature_importance_element_4.append(model.feature_importances_[4])
    quality_feature_importance_element_5.append(model.feature_importances_[5])
    quality_feature_importance_element_6.append(model.feature_importances_[6])    
    # evaluate model
    predictions = model.predict(test[:,:-1])
    score_accuracy = accuracy_score(test[:,-1], predictions)
    print("Accuracy: ", score_accuracy)
    stats_accuracy_quality.append(score_accuracy)
    score_mcc = matthews_corrcoef(test[:,-1], predictions)
    print("MCC: ", score_mcc)
    print("Completion percentage: ", i/n_iterations*100,"%")
    stats_mcc_quality.append(score_mcc)
    
average_quality_feature_importance_element = list()
average_quality_feature_importance_element.append(mean(quality_feature_importance_element_0))
average_quality_feature_importance_element.append(mean(quality_feature_importance_element_1))
average_quality_feature_importance_element.append(mean(quality_feature_importance_element_2))
average_quality_feature_importance_element.append(mean(quality_feature_importance_element_3))
average_quality_feature_importance_element.append(mean(quality_feature_importance_element_4))
average_quality_feature_importance_element.append(mean(quality_feature_importance_element_5))
average_quality_feature_importance_element.append(mean(quality_feature_importance_element_6))


# plot accuracy scores
plt.title("Accuracy Quality for RF")
plt.vlines(mean(stats_accuracy_quality), [0], 3000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_accuracy_quality)))
plt.vlines(median(stats_accuracy_quality), [0], 3000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_accuracy_quality)),color="C1")
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats_accuracy_quality, p))
plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats_accuracy_quality, p))
plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
plt.hist(stats_accuracy_quality, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.legend(loc="upper left")
plt.grid()
plt.show()
# confidence intervals
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats_accuracy_quality, p))
aql = lower
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats_accuracy_quality, p))
aqu = upper
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
  
    
# plot MCC scores
plt.title("MCC Quality for RF")
plt.vlines(mean(stats_mcc_quality), [0], 3000, lw=2.5, linestyle="-", label='Mean = '+"{:.3f}".format(mean(stats_mcc_quality)))
plt.vlines(median(stats_mcc_quality), [0], 3000, lw=2.5, linestyle="-", label='Median = '+"{:.3f}".format(median(stats_mcc_quality)),color="C1")
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats_mcc_quality, p))
plt.vlines(lower, [0], 500, lw=2.5, linestyle="dotted", label=str(alpha*100)+'% CI', color="C2")
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats_mcc_quality, p))
plt.vlines(upper, [0], 500, lw=2.5, linestyle="dotted", color="C2")
plt.hist(stats_mcc_quality, bins=7, color="#0080ff", edgecolor="none", alpha=0.3)
plt.xlabel('MCC')
plt.ylabel('Count')
plt.legend(loc="upper left")
plt.grid()
plt.show()
# confidence intervals
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats_mcc_quality, p))
mql = lower
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats_mcc_quality, p))
mqu = upper
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

feature_names = ['Co Mass','Zn Mass', 
            'CA Mass','KOH Mass', u'H\u2082O Volume', 
            'EtOH Volume', 'Time']


plt.bar(feature_names, average_feature_importance_element)
plt.title("Feature Importance Based on Shape Algorithm")
plt.xticks(fontsize=7)
plt.show()


plt.bar(feature_names, average_quality_feature_importance_element)
plt.title("Feature Importance Based on Quality Algorithm")
plt.xticks(fontsize=7)
plt.show()