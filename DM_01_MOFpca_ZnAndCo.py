# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 22:37:49 2022

@author: Khalid Alotaibi
"""
# Purpose: Conduct principal component analysis (PCA) on the dataset and producing various plots based on the analyzed data
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

# finalDf_transparency = pd.concat([principalDf, pd.DataFrame(Y1_training,columns=['Transparency'])], axis = 1)
# finalDf_transparency = pd.concat([finalDf_transparency, pd.DataFrame(X_training[:,1],columns=['Zn Mass (g)'])], axis = 1)
# finalDf_transparency = pd.concat([finalDf_transparency, pd.DataFrame(X_training[:,5],columns=['EtOH Volume (ml)'])], axis = 1)


# # Predicting the training set 
# # result through scatter plot for Transparency using PCA
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('Transparency 2-Component PCA', fontsize = 20)
# targets = ['Clear', 'Cloudy']
# markers = ['o','^']
# for target, marker in zip(targets,markers):
#     indicesToKeep = finalDf_transparency['Transparency'] == target
#     ax.scatter(finalDf_transparency.loc[indicesToKeep, 'principal component 1']
#                , finalDf_transparency.loc[indicesToKeep, 'principal component 2']
#                , marker = marker
#                , s = 100)
# ax.legend(['Transparent','White-colored'], fontsize=15)
# ax.grid()


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
ax.set_title('MOF Shape 2-Component PCA', fontsize = 20)
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
ax.set_title('MOF Quality 2-Component PCA', fontsize = 20)
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
ax.set_title('Scree Plot of the Component Variances', fontsize=20)
ax.set_xlabel('Principal Component', fontsize=15)
ax.set_ylabel('Explained Variance (%)',fontsize=15)
ax.grid()
plt.grid(axis="x")
plt.show()

# Replace Transparency labels with values using For Loop
for i in range(len(Y1_training)):

	if Y1_training[i] == 'Clear':
		Y1_training[i] = 0
	
	if Y1_training[i] == 'Cloudy':
		Y1_training[i] = 1

Y1_training = Y1_training.astype('int')

for i in range(len(Y1_testing)):

	if Y1_testing[i] == 'Clear':
		Y1_testing[i] = 0
	
	if Y1_testing[i] == 'Cloudy':
		Y1_testing[i] = 1

Y1_testing = Y1_testing.astype('int')

# Replace Hardness labels with values using For Loop
for i in range(len(Y2_training)):

	if Y2_training[i] == 'Y':
		Y2_training[i] = 0
	
	if Y2_training[i] == 'N':
		Y2_training[i] = 1

Y2_training = Y2_training.astype('int')

for i in range(len(Y2_testing)):

	if Y2_testing[i] == 'Y':
		Y2_testing[i] = 0
	
	if Y2_testing[i] == 'N':
		Y2_testing[i] = 1

Y2_testing = Y2_testing.astype('int')


# Replace the "combined" labels with values using For Loop
for i in range(len(Y3_training)):

	if Y3_training[i] == 'Transparent Monolith':
		Y3_training[i] = 0
	
	if Y3_training[i] == 'White-colored Monolith':
		Y3_training[i] = 1
        
	if Y3_training[i] == 'Powder':
		Y3_training[i] = 2
        
Y3_training = list(map(int, Y3_training))

for i in range(len(Y3_testing)):

	if Y3_testing[i] == 'Transparent Monolith':
		Y3_testing[i] = 0
	
	if Y3_testing[i] == 'White-colored Monolith':
		Y3_testing[i] = 1
        
	if Y3_testing[i] == 'Powder':
		Y3_testing[i] = 2

Y3_testing = list(map(int,Y3_testing))

# Visualizing the weight/loading of each feature for PC1 & PC2 & Transparency Values

colormap = ['#A52A2A','r','g','b','y','k','c']

# def myplot(score,coeff,labels=None):
#     xs = score[:,0]
#     ys = score[:,1]
#     n = coeff.shape[0]

#     plt.scatter(xs ,ys, c = Y1_training) #without scaling
#     fig = plt.figure(figsize = (8,8))
#     ax = fig.add_subplot(1,1,1) 
#     ax.set_xlabel('PC 1', fontsize = 15)
#     ax.set_ylabel('PC 2', fontsize = 15)
#     ax.set_title('Transparency PCA Biplot', fontsize = 20)
#     targets = ['Clear', 'Cloudy']
#     markers = ['o','^']
#     for target, marker in zip(targets,markers):
#         indicesToKeep = finalDf_transparency['Transparency'] == target
#         ax.scatter(finalDf_transparency.loc[indicesToKeep, 'principal component 1']
#                , finalDf_transparency.loc[indicesToKeep, 'principal component 2']
#                , marker = marker
#                , s = 100)
#         ax.legend(['Transparent','White-colored'], fontsize=15)
#         ax.grid()
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = colormap[i],alpha = 0.5)
#         if labels is None:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, synthesis_features[i], color = colormap[i], ha = 'center', va = 'center', fontsize=8)
#         else:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

# #Call the function. 
# myplot(principal_components_training_all[:,0:2], pca_all_components. components_) #principal_components_training with pca & principal_component_training_all with pca_all_components
# plt.grid()
# plt.show()

# Visualizing the weight/loading of each feature for PC1 & PC2 & Hardness Values

colormap = ['#A52A2A','r','g','b','y','k','c']

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]

    # plt.scatter(xs ,ys, c = Y2_training) #without scaling
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 2', fontsize = 15)
    ax.set_title('MOF Shape PCA Biplot', fontsize = 20)
    targets = ['Y', 'N']
    markers = ['s','*']
    for target, marker in zip(targets,markers):
        indicesToKeep = finalDf_hardness['Monolith?'] == target
        ax.scatter(finalDf_hardness.loc[indicesToKeep, 'principal component 1']
                   , finalDf_hardness.loc[indicesToKeep, 'principal component 2']
                   , marker = marker
                   , s = 100)
        ax.legend(['Monolith','Powder'],fontsize = 15)
        ax.grid()
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = colormap[i],alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, synthesis_features[i], color = colormap[i], ha = 'center', va = 'center', fontsize=8)
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

#Call the function. 
myplot(principal_components_training_all[:,0:2], pca_all_components. components_) #principal_components_training with pca & principal_component_training_all with pca_all_components
plt.grid()
plt.show()

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]

    # plt.scatter(xs ,ys, c = Y2_training) #without scaling
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 2', fontsize = 15)
    ax.set_title('MOF Quality PCA Biplot', fontsize = 20)
    ax.grid()
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
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = colormap[i],alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, synthesis_features[i], color = colormap[i], ha = 'center', va = 'center', fontsize=8)
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

#Call the function. 
myplot(principal_components_training_all[:,0:2], pca_all_components. components_) #principal_components_training with pca & principal_component_training_all with pca_all_components
plt.grid()
plt.show()

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.title("Scores of Features on PC1 & PC2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()

    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = colormap[i],alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, synthesis_features[i], color = colormap[i], ha = 'center', va = 'center', fontsize=8)
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')


#Call the function. 
myplot(principal_components_training_all[:,0:2], pca_all_components. components_) #principal_components_training with pca & principal_component_training_all with pca_all_components
plt.show()


# # Showing transparency values as a function of Zn mass and EtOH volume
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Zn Mass (g)', fontsize = 15)
# ax.set_ylabel('EtOH Volume (ml)', fontsize = 15)
# ax.set_title('Transparency as a Function of Zn Mass & EtOH Volume', fontsize = 20)
# targets = ['Clear', 'Cloudy']
# markers = ['o','^']
# for target, marker in zip(targets,markers):
#     indicesToKeep = finalDf_transparency['Transparency'] == target
#     ax.scatter(finalDf_transparency.loc[indicesToKeep, 'Zn Mass (g)']
#                , finalDf_transparency.loc[indicesToKeep, 'EtOH Volume (ml)']
#                , marker = marker
#                , s = 100)
# ax.legend(['Transparent','White-colored'])
# ax.grid()



# Showing Hardness values as a function of Zn mass and EtOH volume
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Zn Mass (g)', fontsize = 15)
ax.set_ylabel('EtOH Volume (ml)', fontsize = 15)
ax.set_title('MOF Shape as a Function of Zn Mass & EtOH Volume', fontsize = 20)
targets = ['Y', 'N']
markers = ['s','*']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_hardness['Monolith?'] == target
    ax.scatter(finalDf_hardness.loc[indicesToKeep, 'Zn Mass (g)']
               , finalDf_hardness.loc[indicesToKeep, 'EtOH Volume (ml)']
               , marker = marker
               , s = 100)
ax.legend(['Monolith','Powder'])
ax.grid()

# Showing "Combined" values as a function of Zn mass and EtOH volume
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Zn Mass (g)', fontsize = 15)
ax.set_ylabel('EtOH Volume (ml)', fontsize = 15)
ax.set_title('MOF Quality as a Function of Zn Mass & EtOH Volume', fontsize = 20)
targets = ["Transparent Monolith","White-colored Monolith","Powder"]
markers = ['h','+','1']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_both['MOF Quality'] == target
    ax.scatter(finalDf_both.loc[indicesToKeep, 'Zn Mass (g)']
               , finalDf_both.loc[indicesToKeep, 'EtOH Volume (ml)']
               , marker = marker
               , s = 100)
ax.legend(targets)
ax.grid()

# # Showing Hardness values as a function of water and EtOH volume
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Water Volume (ml)', fontsize = 15)
# ax.set_ylabel('EtOH Volume (ml)', fontsize = 15)
# ax.set_title('MOF Shape as a Function of Water & EtOH Volume', fontsize = 20)
# targets = ['Y', 'N']
# markers = ['s','*']
# for target, marker in zip(targets,markers):
#     indicesToKeep = finalDf_hardness['Monolith?'] == target
#     ax.scatter(finalDf_hardness.loc[indicesToKeep, 'Water Volume (ml)']
#                , finalDf_hardness.loc[indicesToKeep, 'EtOH Volume (ml)']
#                , marker = marker
#                , s = 100)
# ax.legend(['Monolith','Powder'])
# ax.grid()


# # Showing "Combined" values as a function of Water Volume and EtOH volume
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Water Volume (ml)', fontsize = 15)
# ax.set_ylabel('EtOH Volume (ml)', fontsize = 15)
# ax.set_title('MOF Quality as a Function of Water & EtOH Volume', fontsize = 20)
# targets = ["Transparent Monolith","White-colored Monolith","Powder"]
# markers = ['h','+','1']
# for target, marker in zip(targets,markers):
#     indicesToKeep = finalDf_both['MOF Quality'] == target
#     ax.scatter(finalDf_both.loc[indicesToKeep, 'Water Volume (ml)']
#                , finalDf_both.loc[indicesToKeep, 'EtOH Volume (ml)']
#                , marker = marker
#                , s = 100)
# ax.legend(targets)
# ax.grid()

print(pca_all_components.explained_variance_)

import pandas as pd
from factor_analyzer import FactorAnalyzer
import numpy as np
import matplotlib.pyplot as plt

def _HornParallelAnalysis(data, K=10, printEigenvalues=False):
    ################
    # Create a random matrix to match the dataset
    ################
    n, m = data.shape
    # Set the factor analysis parameters
    fa = FactorAnalyzer(n_factors=1, method='minres', rotation=None, use_smc=True)
    # Create arrays to store the values
    sumComponentEigens = np.empty(m)
    sumFactorEigens = np.empty(m)
    # Run the fit 'K' times over a random matrix
    for runNum in range(0, K):
        fa.fit(np.random.normal(size=(n, m)))
        sumComponentEigens = sumComponentEigens + fa.get_eigenvalues()[0]
        sumFactorEigens = sumFactorEigens + fa.get_eigenvalues()[1]
    # Average over the number of runs
    avgComponentEigens = sumComponentEigens / K
    avgFactorEigens = sumFactorEigens / K

    ################
    # Get the eigenvalues for the fit on supplied data
    ################
    fa.fit(data)
    dataEv = fa.get_eigenvalues()
    # Set up a scree plot
    plt.figure(figsize=(8, 6))

    ################
    ### Print results
    ################
    if printEigenvalues:
        print('Principal component eigenvalues for random matrix:\n', avgComponentEigens)
        print('Factor eigenvalues for random matrix:\n', avgFactorEigens)
        print('Principal component eigenvalues for data:\n', dataEv[0])
        print('Factor eigenvalues for data:\n', dataEv[1])
    # Find the suggested stopping points
    suggestedFactors = sum((dataEv[1] - avgFactorEigens) > 0)
    suggestedComponents = sum((dataEv[0] - avgComponentEigens) > 0)
    print('Parallel analysis suggests that the number of factors = ', suggestedFactors , ' and the number of components = ', suggestedComponents)


    ################
    ### Plot the eigenvalues against the number of variables
    ################
    # Line for eigenvalue 1
    plt.plot([0, m+1], [1, 1], 'k--', alpha=0.3)
    # For the random data - Components
    plt.plot(range(1, m+1), avgComponentEigens, 'b', label='PC - random', alpha=0.4)
    # For the Data - Components
    plt.scatter(range(1, m+1), dataEv[0], c='b', marker='o')
    plt.plot(range(1, m+1), dataEv[0], 'b', label='PC - data')
    # For the random data - Factors
    plt.plot(range(1, m+1), avgFactorEigens, 'g', label='FA - random', alpha=0.4)
    # For the Data - Factors
    plt.scatter(range(1, m+1), dataEv[1], c='g', marker='o')
    plt.plot(range(1, m+1), dataEv[1], 'g', label='FA - data')
    plt.title('Parallel Analysis Scree Plots', {'fontsize': 20})
    plt.xlabel('Factors/Components', {'fontsize': 15})
    plt.xticks(ticks=range(1, m+1), labels=range(1, m+1))
    plt.ylabel('Eigenvalue', {'fontsize': 15})
    plt.legend()
    plt.show();
    

_HornParallelAnalysis(principal_components_training_all)