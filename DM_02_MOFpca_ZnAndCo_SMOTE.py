# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 22:37:49 2022

@author: Khalid Alotaibi
"""
# Purpose: Conduct principal component analysis (PCA) on the dataset and produce various plots based on the analyzed data. If data augmentation is applied, the applied technique is reflected on the file title.
####################################################################

# Step 1: Import needed libraries   
import numpy as np
import pandas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import BorderlineSMOTE

# Step 2: Import Dataset
RandForDataset = pandas.read_csv("UTSA_16_A2ML_Data_Updated.csv")

Dataset_zn = pandas.read_csv("UTSA_16_A2ML_Data_Updated.csv")

Dataset_co = pandas.read_csv("UTSA_16_A2ML_Data_Updated.csv")
Dataset_co = Dataset_co.replace(0, pd.np.nan).dropna(axis=0, how='any', subset=['Cobalt Acetate Co(C2H3O2)2 Mass (g)']).fillna(0)



#print("The column headers :")
#print(list(RandForDataset.columns.values))
Y1 = np.array(Dataset_zn['Monolith Transparency\nClear - Glass-looking\nCloudy - White-colored']);
Y2 = np.array(Dataset_zn['Monolith?']);
Y3 = [0]*len(Y2)
number_of_datapoints = len(Y1)
for i in range(number_of_datapoints):
    if Y1[i] == 'Clear' and Y2[i] == 'Y':
        Y3[i] = 'Transparent Monolith'
    if Y1[i] == 'Cloudy' and Y2[i] == 'Y':
        Y3[i] = 'White-colored Monolith'
    if Y2[i] == 'N':
        Y3[i] = 'Powder'
        
condlist_zn = ['Zinc Acetate Zn(CH3CO2)2 Mass (g)', 
            'Citric Acid HOC(CO2H)(CH2CO2H)2 Mass (g)', 
            'Potassium Hydroxide KOH Mass (g)', 'Water Volume (ml)', 
            'Ethanol EtOH Volume (ml)', 'Synthesis Time (hr)']

condlist_co = ['Cobalt Acetate Co(C2H3O2)2 Mass (g)', 
            'Citric Acid HOC(CO2H)(CH2CO2H)2 Mass (g)', 
            'Potassium Hydroxide KOH Mass (g)', 'Water Volume (ml)', 
            'Ethanol EtOH Volume (ml)', 'Synthesis Time (hr)']

condlist_znco = ['Cobalt Acetate Co(C2H3O2)2 Mass (g)','Zinc Acetate Zn(CH3CO2)2 Mass (g)', 
            'Citric Acid HOC(CO2H)(CH2CO2H)2 Mass (g)', 
            'Potassium Hydroxide KOH Mass (g)', 'Water Volume (ml)', 
            'Ethanol EtOH Volume (ml)', 'Synthesis Time (hr)']

# Create the training and testing sets
for train_index, test_index in StratifiedShuffleSplit(n_splits=2,test_size=0.2,random_state=16).split(Dataset_zn[condlist_znco],Dataset_zn['Monolith Transparency\nClear - Glass-looking\nCloudy - White-colored']):
    train = Dataset_zn.loc[train_index,:]
    test = Dataset_zn.loc[test_index,:]

training_data_zn = train
testing_data_zn = test


X_training = np.array(training_data_zn[condlist_znco])
Y1_training = np.array(training_data_zn['Monolith Transparency\nClear - Glass-looking\nCloudy - White-colored']);
Y2_training = np.array(training_data_zn['Monolith?'])

X_testing = np.array(testing_data_zn[condlist_znco])
Y1_testing = np.array(testing_data_zn['Monolith Transparency\nClear - Glass-looking\nCloudy - White-colored']);
Y2_testing = np.array(testing_data_zn['Monolith?'])

# Define the number of variables/properties
number_of_variables = len(X_training[0])
number_of_training_datapoints = len(X_training)
number_of_testing_datapoints = len(X_testing)


Y3 = [0]*len(Y2)
number_of_datapoints = len(Y1)
for i in range(number_of_datapoints):
    if Y1[i] == 'Clear' and Y2[i] == 'Y':
        Y3[i] = 'Transparent Monolith'
    if Y1[i] == 'Cloudy' and Y2[i] == 'Y':
        Y3[i] = 'White-colored Monolith'
    if Y2[i] == 'N':
        Y3[i] = 'Powder'

synthesis_features = ['Co Mass','Zn Mass', 
            'Acid Mass', 
            'Base Mass', 'Water Volume', 
            'EtOH Volume', 'Synthesis Time']

# Define the number of variables/properties
number_of_variables = len(X_training[0])
number_of_training_datapoints = len(X_training)
number_of_testing_datapoints = len(X_testing)

# Add the new labels combining both transparency and hardness values
Y3_training = [0]*number_of_training_datapoints
Y3_testing = [0]*number_of_testing_datapoints
for i in range(number_of_training_datapoints):
    if Y1_training[i] == 'Clear' and Y2_training[i] == 'Y':
        Y3_training[i] = 'Transparent Monolith'
    if Y1_training[i] == 'Cloudy' and Y2_training[i] == 'Y':
        Y3_training[i] = 'White-colored Monolith'
    if Y2_training[i] == 'N':
        Y3_training[i] = 'Powder'

for i in range(number_of_testing_datapoints):
    if Y1_testing[i] == 'Clear' and Y2_testing[i] == 'Y':
        Y3_testing[i] = 'Transparent Monolith'
    if Y1_testing[i] == 'Cloudy' and Y2_testing[i] == 'Y':
        Y3_testing[i] = 'White-colored Monolith'
    if Y2_testing[i] == 'N':
        Y3_testing[i] = 'Powder'

# performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_training_scaled = sc.fit_transform(X_training)
X_testing_scaled = sc.transform(X_testing)

# Applying PCA function on training & testing sets 
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_all_components = PCA()

principal_components_training = pca.fit_transform(X_training_scaled)
principal_components_testing = pca.transform(X_testing_scaled)
principal_components_training_all = pca_all_components.fit_transform(X_training_scaled)
principal_components_testing_all = pca_all_components.transform(X_testing_scaled)

explained_variance_pca = pca.explained_variance_ratio_*100
explained_variance_pca_all_components = pca_all_components.explained_variance_ratio_*100

principalDf = pd.DataFrame(data = principal_components_training
             , columns = ['principal component 1', 'principal component 2'])

# Define a pipeline
oversample = BorderlineSMOTE(k_neighbors=1,random_state=2)
oversampled_PC_shape = oversample.fit_resample(principal_components_training_all, Y2_training)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('MOF Shape 2-Component PCA with SMOTE', fontsize = 20)
targets = ['Y', 'N']
markers = ['s','*']
for target, marker in zip(targets,markers):
    indicesToKeep = oversampled_PC_shape[1] == target
    ax.scatter(oversampled_PC_shape[0][indicesToKeep, 0]
               , oversampled_PC_shape[0][indicesToKeep, 1]
               , marker = marker
               , s = 100)
ax.legend(['Monolith','Powder'], fontsize=15)
ax.grid()


# Convert target variables from 'str' to 'int'
from sklearn.preprocessing import LabelEncoder
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()
encoded_y1 = label_encoder1.fit_transform(Y1)
encoded_y2 = label_encoder2.fit_transform(Y2_training)
encoded_y3 = label_encoder3.fit_transform(Y3_training)
label_encoder_name_mapping1 = dict(zip(label_encoder1.classes_,label_encoder1.transform(label_encoder1.classes_)))
label_encoder_name_mapping2 = dict(zip(label_encoder2.classes_,label_encoder2.transform(label_encoder2.classes_)))
label_encoder_name_mapping3 = dict(zip(label_encoder3.classes_,label_encoder3.transform(label_encoder3.classes_)))



data_shape = np.column_stack((principal_components_training_all,encoded_y2))
data_quality = np.column_stack((principal_components_training_all,encoded_y3))

oversampled_PC_quality = oversample.fit_resample(principal_components_training_all, encoded_y3)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('MOF Quality 2-Component PCA  with SMOTE', fontsize = 20)
targets = [1, 2,0]
markers = ['h','+','1']
for target, marker in zip(targets,markers):
    indicesToKeep = oversampled_PC_quality[1] == target
    ax.scatter(oversampled_PC_quality[0][indicesToKeep, 0]
               , oversampled_PC_quality[0][indicesToKeep, 1]
               , marker = marker
               , s = 100)
ax.legend(['Transparent Monolith','White-colored Monolith','Powder'], fontsize=15)
ax.grid()

