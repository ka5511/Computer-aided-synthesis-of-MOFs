# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:28:39 2022

@author: Khalid Alotaibi
"""
# Purpose: Perform bootstrap sampling to train a random forest algorithm (either base or optimized version) and check MOF shape/quality prediction performance. Input to the model consists of only the first 3 principal components (PC) obtained from the PCA analysis. If data augmentation is applied, the applied technique is reflected on the file title.
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

data_shape = np.column_stack((principal_components,encoded_y2))
data_quality = np.column_stack((principal_components,encoded_y3))


# configure bootstrap with the number of sample sets and the size of each sample
n_iterations = 10000
n_size = int(len(data_shape) * 1)
# run bootstrap for unoptimized model
stats_accuracy_shape = list()
stats_mcc_shape = list()
for i in range(n_iterations):
    # prepare train and test sets
    train = resample(data_shape, n_samples=n_size,stratify=data_shape[:,-1])
    test = np.array([x for x in data_shape if x.tolist() not in train.tolist()])
    # fit model
    model = RandomForestClassifier()
    model.fit(train[:,:-1], train[:,-1])
    # evaluate model
    predictions = model.predict(test[:,:-1])
    score_accuracy = accuracy_score(test[:,-1], predictions)
    print("Accuracy: ", score_accuracy)
    stats_accuracy_shape.append(score_accuracy)
    score_mcc = matthews_corrcoef(test[:,-1], predictions)
    print("MCC: ", score_mcc)
    print("Completion percentage: ", i/n_iterations*100,"%")
    stats_mcc_shape.append(score_mcc)

# plot accuracy scores
plt.title("Accuracy Shape")
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
plt.title("MCC Shape")
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


# run bootstrap for unoptimized model
stats_accuracy_quality = list()
stats_mcc_quality = list()
for i in range(n_iterations):
    # prepare train and test sets
    train = resample(data_quality, n_samples=n_size,stratify=data_quality[:,-1])
    test = np.array([x for x in data_quality if x.tolist() not in train.tolist()])
    # fit model
    model = RandomForestClassifier()
    model.fit(train[:,:-1], train[:,-1])
    # evaluate model
    predictions = model.predict(test[:,:-1])
    score_accuracy = accuracy_score(test[:,-1], predictions)
    print("Accuracy: ", score_accuracy)
    stats_accuracy_quality.append(score_accuracy)
    score_mcc = matthews_corrcoef(test[:,-1], predictions)
    print("MCC: ", score_mcc)
    print("Completion percentage: ", i/n_iterations*100,"%")
    stats_mcc_quality.append(score_mcc)

# plot accuracy scores
plt.title("Accuracy Quality")
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
plt.title("MCC Quality")
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