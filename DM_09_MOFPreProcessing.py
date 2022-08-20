# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 22:37:49 2022

@author: Khalid Alotaibi
"""
# Purpose: Compare different preprocessing techniques on MOF shape/quality data to see which one transforms the data into a format that is more easily and effectively processed by classification algorithms
####################################################################

# Step 1: Import needed libraries   
import numpy as np
import pandas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

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

# performing preprocessing partc ########################### STANDARD SCALER
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

# Predicting the training set 
# result through scatter plot for Hardness using PCA
finalDf_hardness = pd.concat([principalDf, pd.DataFrame(Y2_training,columns=['Monolith?'])], axis = 1)
finalDf_hardness = pd.concat([finalDf_hardness, pd.DataFrame(X_training[:,1],columns=['Zn Mass (g)'])], axis = 1)
finalDf_hardness = pd.concat([finalDf_hardness, pd.DataFrame(X_training[:,5],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_hardness = pd.concat([finalDf_hardness, pd.DataFrame(X_training[:,4],columns=['Water Volume (ml)'])], axis = 1)



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('MOF Shape 2-Component PCA - Standard Scaler', fontsize = 20)
targets = ['Y', 'N']
markers = ['s','*']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_hardness['Monolith?'] == target
    ax.scatter(finalDf_hardness.loc[indicesToKeep, 'principal component 1']
               , finalDf_hardness.loc[indicesToKeep, 'principal component 2']
               , marker = marker
               , s = 100)
ax.legend(['Monolith','Powder'], fontsize=15)
ax.grid()

# Predicting the training set 
# result through scatter plot for the 'Combined' label using PCA
finalDf_both = pd.concat([principalDf, pd.DataFrame(Y3_training,columns=['MOF Quality'])], axis = 1)
finalDf_both = pd.concat([finalDf_both, pd.DataFrame(X_training[:,1],columns=['Zn Mass (g)'])], axis = 1)
finalDf_both = pd.concat([finalDf_both, pd.DataFrame(X_training[:,5],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_both = pd.concat([finalDf_both, pd.DataFrame(X_training[:,4],columns=['Water Volume (ml)'])], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('MOF Quality 2-Component PCA - Standard Scaler', fontsize = 20)
targets = ['Transparent Monolith', 'White-colored Monolith','Powder']
markers = ['h','+','1']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_both['MOF Quality'] == target
    ax.scatter(finalDf_both.loc[indicesToKeep, 'principal component 1']
               , finalDf_both.loc[indicesToKeep, 'principal component 2']
               , marker = marker
               , s = 100)
ax.legend(targets, fontsize=15)
ax.grid()


# Visualizing the explained variance represented by each PC
fig = plt.figure(figsize=(8,8))
plt.bar(['PC1','PC2','PC3','PC4','PC5','PC6','PC7'],explained_variance_pca_all_components)
ax = fig.add_subplot(1,1,1)
ax.set_title('Scree Plot of the Component Variances - Standard Scaler', fontsize=20)
ax.set_xlabel('Principal Component', fontsize=15)
ax.set_ylabel('Explained Variance (%)',fontsize=15)
ax.grid()
plt.grid(axis="x")
plt.show()

# performing preprocessing partc ########################### Robust Scaler
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()

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

# Predicting the training set 
# result through scatter plot for Hardness using PCA
finalDf_hardness = pd.concat([principalDf, pd.DataFrame(Y2_training,columns=['Monolith?'])], axis = 1)
finalDf_hardness = pd.concat([finalDf_hardness, pd.DataFrame(X_training[:,1],columns=['Zn Mass (g)'])], axis = 1)
finalDf_hardness = pd.concat([finalDf_hardness, pd.DataFrame(X_training[:,5],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_hardness = pd.concat([finalDf_hardness, pd.DataFrame(X_training[:,4],columns=['Water Volume (ml)'])], axis = 1)



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('MOF Shape 2-Component PCA - Robust Scaler', fontsize = 20)
targets = ['Y', 'N']
markers = ['s','*']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_hardness['Monolith?'] == target
    ax.scatter(finalDf_hardness.loc[indicesToKeep, 'principal component 1']
               , finalDf_hardness.loc[indicesToKeep, 'principal component 2']
               , marker = marker
               , s = 100)
ax.legend(['Monolith','Powder'], fontsize=15)
ax.grid()

# Predicting the training set 
# result through scatter plot for the 'Combined' label using PCA
finalDf_both = pd.concat([principalDf, pd.DataFrame(Y3_training,columns=['MOF Quality'])], axis = 1)
finalDf_both = pd.concat([finalDf_both, pd.DataFrame(X_training[:,1],columns=['Zn Mass (g)'])], axis = 1)
finalDf_both = pd.concat([finalDf_both, pd.DataFrame(X_training[:,5],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_both = pd.concat([finalDf_both, pd.DataFrame(X_training[:,4],columns=['Water Volume (ml)'])], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('MOF Quality 2-Component PCA - Robust Scaler', fontsize = 20)
targets = ['Transparent Monolith', 'White-colored Monolith','Powder']
markers = ['h','+','1']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_both['MOF Quality'] == target
    ax.scatter(finalDf_both.loc[indicesToKeep, 'principal component 1']
               , finalDf_both.loc[indicesToKeep, 'principal component 2']
               , marker = marker
               , s = 100)
ax.legend(targets, fontsize=15)
ax.grid()


# Visualizing the explained variance represented by each PC
fig = plt.figure(figsize=(8,8))
plt.bar(['PC1','PC2','PC3','PC4','PC5','PC6','PC7'],explained_variance_pca_all_components)
ax = fig.add_subplot(1,1,1)
ax.set_title('Scree Plot of the Component Variances - Robust Scaler', fontsize=20)
ax.set_xlabel('Principal Component', fontsize=15)
ax.set_ylabel('Explained Variance (%)',fontsize=15)
ax.grid()
plt.grid(axis="x")
plt.show()


# performing preprocessing partc ########################### Normalizer
from sklearn.preprocessing import Normalizer
sc = Normalizer()

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

# Predicting the training set 
# result through scatter plot for Hardness using PCA
finalDf_hardness = pd.concat([principalDf, pd.DataFrame(Y2_training,columns=['Monolith?'])], axis = 1)
finalDf_hardness = pd.concat([finalDf_hardness, pd.DataFrame(X_training[:,1],columns=['Zn Mass (g)'])], axis = 1)
finalDf_hardness = pd.concat([finalDf_hardness, pd.DataFrame(X_training[:,5],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_hardness = pd.concat([finalDf_hardness, pd.DataFrame(X_training[:,4],columns=['Water Volume (ml)'])], axis = 1)



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('MOF Shape 2-Component PCA - Normalizer', fontsize = 20)
targets = ['Y', 'N']
markers = ['s','*']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_hardness['Monolith?'] == target
    ax.scatter(finalDf_hardness.loc[indicesToKeep, 'principal component 1']
               , finalDf_hardness.loc[indicesToKeep, 'principal component 2']
               , marker = marker
               , s = 100)
ax.legend(['Monolith','Powder'], fontsize=15)
ax.grid()

# Predicting the training set 
# result through scatter plot for the 'Combined' label using PCA
finalDf_both = pd.concat([principalDf, pd.DataFrame(Y3_training,columns=['MOF Quality'])], axis = 1)
finalDf_both = pd.concat([finalDf_both, pd.DataFrame(X_training[:,1],columns=['Zn Mass (g)'])], axis = 1)
finalDf_both = pd.concat([finalDf_both, pd.DataFrame(X_training[:,5],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_both = pd.concat([finalDf_both, pd.DataFrame(X_training[:,4],columns=['Water Volume (ml)'])], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('MOF Quality 2-Component PCA - Normalizer', fontsize = 20)
targets = ['Transparent Monolith', 'White-colored Monolith','Powder']
markers = ['h','+','1']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_both['MOF Quality'] == target
    ax.scatter(finalDf_both.loc[indicesToKeep, 'principal component 1']
               , finalDf_both.loc[indicesToKeep, 'principal component 2']
               , marker = marker
               , s = 100)
ax.legend(targets, fontsize=15)
ax.grid()


# Visualizing the explained variance represented by each PC
fig = plt.figure(figsize=(8,8))
plt.bar(['PC1','PC2','PC3','PC4','PC5','PC6','PC7'],explained_variance_pca_all_components)
ax = fig.add_subplot(1,1,1)
ax.set_title('Scree Plot of the Component Variances - Normalizer', fontsize=20)
ax.set_xlabel('Principal Component', fontsize=15)
ax.set_ylabel('Explained Variance (%)',fontsize=15)
ax.grid()
plt.grid(axis="x")
plt.show()