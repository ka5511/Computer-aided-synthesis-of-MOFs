# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:54:05 2022

@author: Khalid Alotaibi
"""
# Purpose: Obtain regression coefficients for the best performing models in terms of predicting the quantitative MOF synthesis results. In addition, store best performing models in .joblib files.
####################################################################
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
from matplotlib.ticker import FormatStrFormatter

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

std_dataframe = RandForDataset[synthesis_features].std(axis=0)
std_values = np.array(std_dataframe)
X = np.array(RandForDataset[synthesis_features])
Y = np.array(RandForDataset[synthesis_outputs])
data_density = np.column_stack((X,Y[:,0]))
data_size = np.column_stack((X,Y[:,1]))
data_BET = np.column_stack((X,Y[:,2]))
data_micro_volume = np.column_stack((X,Y[:,3]))
data_total_volume = np.column_stack((X,Y[:,4]))
# Save best models
opt_model = Ridge(random_state=0).fit(X,Y[:,0])
print(opt_model.predict(X))
filename2 = "Density_Model.joblib"
joblib.dump(opt_model, filename2)

opt_model = Ridge(random_state=0).fit(X,Y[:,1])
print(opt_model.predict(X))
filename2 = "Size_Model.joblib"
joblib.dump(opt_model, filename2)

opt_model = Ridge(random_state=0).fit(X,Y[:,2])
print(opt_model.predict(X))
filename2 = "BET_Model.joblib"
joblib.dump(opt_model, filename2)

opt_model = Ridge(random_state=0).fit(X,Y[:,3])
print(opt_model.predict(X))
filename2 = "Micro_Model.joblib"
joblib.dump(opt_model, filename2)

opt_model = Ridge(random_state=0).fit(X,Y[:,4])
print(opt_model.predict(X))
filename2 = "Total_Volume_Model.joblib"
joblib.dump(opt_model, filename2)



def get_models_density():
    models = list()
    # models.append(Ridge(random_state=0))
    return models

def get_models_size():
    models = list()
    # models.append(Ridge(random_state=0))
    return models

def get_models_BET():
    models = list()
    # models.append(Ridge(random_state=0))
    return models

def get_models_micro_volume():
    models = list()
    # models.append(Ridge(random_state=0))
    return models


def get_models_total_volume():
    models = list()
    # models.append(Ridge(random_state=0))
    return models

# models_list = ['RF','LR','Ridge','Lasso','KNN','SVR','Meta]

# models_list_expanded = ['Random Forest','Linear Regression','Ridge',
#                         'Lasso','KNN','Support Vector','Meta']

models_outputs = ['Density', 'Particle Size', 'BET Area', 'Micropore Volume','Total Pore Volume']

# K-fold cross-validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

# performing preprocessing part to scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
principal_components = pca.fit_transform(X_scaled)

# Import the optimized models
density_model = Ridge(random_state=0)
size_model = Ridge(random_state=0)
BET_model = Ridge(random_state=0)
micro_model = Ridge(random_state=0)
total_model = Ridge(random_state=0)

# Store results of all models for all outputs in lists 
density_feature_importance_element_0= list()
density_feature_importance_element_1= list()
density_feature_importance_element_2= list()
density_feature_importance_element_3= list()
density_feature_importance_element_4= list()
density_feature_importance_element_5= list()

# configure bootstrap with the number of sample sets and the size of each sample
n_iterations = 1000
n_size = int(len(data_density) * 0.8)
for i in range(n_iterations):
    # prepare train and test sets
    train = resample(data_density, n_samples=n_size)
    test = np.array([x for x in data_density if x.tolist() not in train.tolist()])
    # fit model
    model = density_model.fit(train[:,:-1], train[:,-1])
    density_feature_importance_element_0.append(model.coef_[0])
    density_feature_importance_element_1.append(model.coef_[1])
    density_feature_importance_element_2.append(model.coef_[2])
    density_feature_importance_element_3.append(model.coef_[3])
    density_feature_importance_element_4.append(model.coef_[4])
    density_feature_importance_element_5.append(model.coef_[5])
    print("Completion percentage: ", i/n_iterations*100,"%")

density_average_feature_importance_element = list()
density_average_feature_importance_element.append(mean(density_feature_importance_element_0))
density_average_feature_importance_element.append(mean(density_feature_importance_element_1))
density_average_feature_importance_element.append(mean(density_feature_importance_element_2))
density_average_feature_importance_element.append(mean(density_feature_importance_element_3))
density_average_feature_importance_element.append(mean(density_feature_importance_element_4))
density_average_feature_importance_element.append(mean(density_feature_importance_element_5))
  
# Size calculations    
size_feature_importance_element_0= list()
size_feature_importance_element_1= list()
size_feature_importance_element_2= list()
size_feature_importance_element_3= list()
size_feature_importance_element_4= list()
size_feature_importance_element_5= list()

for i in range(n_iterations):
    # prepare train and test sets
    train = resample(data_size, n_samples=n_size)
    test = np.array([x for x in data_size if x.tolist() not in train.tolist()])
    # fit model
    model= size_model.fit(train[:,:-1], train[:,-1])
    size_feature_importance_element_0.append(model.coef_[0])
    size_feature_importance_element_1.append(model.coef_[1])
    size_feature_importance_element_2.append(model.coef_[2])
    size_feature_importance_element_3.append(model.coef_[3])
    size_feature_importance_element_4.append(model.coef_[4])
    size_feature_importance_element_5.append(model.coef_[5])
    print("Completion percentage: ", i/n_iterations*100,"%")
    
size_average_feature_importance_element = list()
size_average_feature_importance_element.append(mean(size_feature_importance_element_0))
size_average_feature_importance_element.append(mean(size_feature_importance_element_1))
size_average_feature_importance_element.append(mean(size_feature_importance_element_2))
size_average_feature_importance_element.append(mean(size_feature_importance_element_3))
size_average_feature_importance_element.append(mean(size_feature_importance_element_4))
size_average_feature_importance_element.append(mean(size_feature_importance_element_5))
  

# BET calculations    
BET_feature_importance_element_0= list()
BET_feature_importance_element_1= list()
BET_feature_importance_element_2= list()
BET_feature_importance_element_3= list()
BET_feature_importance_element_4= list()
BET_feature_importance_element_5= list()

for i in range(n_iterations):
    # prepare train and test sets
    train = resample(data_BET, n_samples=n_size)
    test = np.array([x for x in data_BET if x.tolist() not in train.tolist()])
    # fit model
    model= BET_model.fit(train[:,:-1], train[:,-1])
    BET_feature_importance_element_0.append(model.coef_[0])
    BET_feature_importance_element_1.append(model.coef_[1])
    BET_feature_importance_element_2.append(model.coef_[2])
    BET_feature_importance_element_3.append(model.coef_[3])
    BET_feature_importance_element_4.append(model.coef_[4])
    BET_feature_importance_element_5.append(model.coef_[5])
    print("Completion percentage: ", i/n_iterations*100,"%")
    
BET_average_feature_importance_element = list()
BET_average_feature_importance_element.append(mean(BET_feature_importance_element_0))
BET_average_feature_importance_element.append(mean(BET_feature_importance_element_1))
BET_average_feature_importance_element.append(mean(BET_feature_importance_element_2))
BET_average_feature_importance_element.append(mean(BET_feature_importance_element_3))
BET_average_feature_importance_element.append(mean(BET_feature_importance_element_4))
BET_average_feature_importance_element.append(mean(BET_feature_importance_element_5))
  

# Micro Volume calculations    
micro_feature_importance_element_0= list()
micro_feature_importance_element_1= list()
micro_feature_importance_element_2= list()
micro_feature_importance_element_3= list()
micro_feature_importance_element_4= list()
micro_feature_importance_element_5= list()


for i in range(n_iterations):
    # prepare train and test sets
    train = resample(data_micro_volume, n_samples=n_size)
    test = np.array([x for x in data_micro_volume if x.tolist() not in train.tolist()])
    # fit model
    model = micro_model.fit(train[:,:-1], train[:,-1])
    micro_feature_importance_element_0.append(model.coef_[0])
    micro_feature_importance_element_1.append(model.coef_[1])
    micro_feature_importance_element_2.append(model.coef_[2])
    micro_feature_importance_element_3.append(model.coef_[3])
    micro_feature_importance_element_4.append(model.coef_[4])
    micro_feature_importance_element_5.append(model.coef_[5])
    print("Completion percentage: ", i/n_iterations*100,"%")
    
micro_average_feature_importance_element = list()
micro_average_feature_importance_element.append(mean(micro_feature_importance_element_0))
micro_average_feature_importance_element.append(mean(micro_feature_importance_element_1))
micro_average_feature_importance_element.append(mean(micro_feature_importance_element_2))
micro_average_feature_importance_element.append(mean(micro_feature_importance_element_3))
micro_average_feature_importance_element.append(mean(micro_feature_importance_element_4))
micro_average_feature_importance_element.append(mean(micro_feature_importance_element_5))




# Total Volume calculations    
total_feature_importance_element_0= list()
total_feature_importance_element_1= list()
total_feature_importance_element_2= list()
total_feature_importance_element_3= list()
total_feature_importance_element_4= list()
total_feature_importance_element_5= list()
for i in range(n_iterations):
    # prepare train and test sets
    train = resample(data_total_volume, n_samples=n_size)
    test = np.array([x for x in data_total_volume if x.tolist() not in train.tolist()])
    # fit model
    model = total_model.fit(train[:,:-1], train[:,-1])
    total_feature_importance_element_0.append(model.coef_[0])
    total_feature_importance_element_1.append(model.coef_[1])
    total_feature_importance_element_2.append(model.coef_[2])
    total_feature_importance_element_3.append(model.coef_[3])
    total_feature_importance_element_4.append(model.coef_[4])
    total_feature_importance_element_5.append(model.coef_[5])
    print("Completion percentage: ", i/n_iterations*100,"%")

total_average_feature_importance_element = list()
total_average_feature_importance_element.append(mean(total_feature_importance_element_0))
total_average_feature_importance_element.append(mean(total_feature_importance_element_1))
total_average_feature_importance_element.append(mean(total_feature_importance_element_2))
total_average_feature_importance_element.append(mean(total_feature_importance_element_3))
total_average_feature_importance_element.append(mean(total_feature_importance_element_4))
total_average_feature_importance_element.append(mean(total_feature_importance_element_5))


feature_names = ['Zn Mass', 
            'CA Mass','KOH Mass', u'H\u2082O Volume', 
            'EtOH Volume', 'Time']

fig, ax = plt.subplots()
plt.barh(feature_names, density_average_feature_importance_element)
plt.figsize=(6, 6)
plt.title("Raw Coefficients Based on Density Algorithm")
plt.xlabel('Raw Coefficient Values')
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
plt.show()

fig, ax = plt.subplots()
plt.barh(feature_names, size_average_feature_importance_element)
plt.figsize=(6, 6)
plt.title("Raw Coefficients Based on Size Algorithm")
plt.xlabel('Raw Coefficient Values')
plt.axvline(x=0, color=".5")
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.subplots_adjust(left=0.3)
plt.show()

fig, ax = plt.subplots()
plt.barh(feature_names, BET_average_feature_importance_element)
plt.figsize=(6, 6)
plt.title("Raw Coefficients Based on BET Algorithm")
plt.xlabel('Raw Coefficient Values')
plt.axvline(x=0, color=".5")
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.subplots_adjust(left=0.3)
plt.show()

fig, ax = plt.subplots()
plt.barh(feature_names, micro_average_feature_importance_element)
plt.figsize=(6, 6)
plt.title("Raw Coefficients Based on Micropore Volume Algorithm")
plt.xlabel('Raw Coefficient Values')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.axvline(x=0, color=".5")
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.subplots_adjust(left=0.3)
plt.show()

fig, ax = plt.subplots()
plt.barh(feature_names, total_average_feature_importance_element)
plt.figsize=(6, 6)
plt.title("Raw Coefficients Based on Total Pore Volume Algorithm")
plt.xlabel('Raw Coefficient Values')
plt.axvline(x=0, color=".5")
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.subplots_adjust(left=0.3)
plt.show()



# Coefficient values corrected/scaled by the feature's std. dev.
fig, ax = plt.subplots()
plt.barh(feature_names, std_values*density_average_feature_importance_element)
plt.title("Corrected Coefficients Based on Density Algorithm")
plt.xlabel('Scaled Coefficient Values')
plt.figsize=(6, 6)
plt.axvline(x=0, color=".5")
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.subplots_adjust(left=0.3)
plt.show()


fig, ax = plt.subplots()
plt.barh(feature_names, std_values*size_average_feature_importance_element)
plt.title("Corrected Coefficients Based on Size Algorithm")
plt.xlabel('Scaled Coefficient Values')
plt.figsize=(6, 6)
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
plt.show()

fig, ax = plt.subplots()
plt.barh(feature_names, std_values*BET_average_feature_importance_element)
plt.title("Corrected Coefficients Based on BET Algorithm")
plt.xlabel('Scaled Coefficient Values')
plt.figsize=(6, 6)
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.show()

fig, ax = plt.subplots()
plt.barh(feature_names, std_values*micro_average_feature_importance_element)
plt.title("Corrected Coefficients Based on Micropore Volume Algorithm")
plt.xlabel('Scaled Coefficient Values')
plt.figsize=(6, 6)
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.axvline(x=0, color=".5")
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.subplots_adjust(left=0.3)
plt.show()

fig, ax = plt.subplots()
plt.barh(feature_names, std_values*total_average_feature_importance_element)
plt.title("Corrected Coefficients Based on Total Pore Volume Algorithm")
plt.xlabel('Scaled Coefficient Values')
plt.figsize=(6, 6)
xabs_max = abs(max(ax.get_xlim(), key=abs))
ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)
plt.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
plt.show()


# Do single plots for EtOH, H2O, and time
feature_importance = list()
feature_importance.append(density_average_feature_importance_element[4])
feature_importance.append(size_average_feature_importance_element[4])
feature_importance.append(BET_average_feature_importance_element[4])
feature_importance.append(micro_average_feature_importance_element[4])
feature_importance.append(total_average_feature_importance_element[4])

models_outputs2 = list()
models_outputs2.append(models_outputs[0])
models_outputs2.append(models_outputs[3])
models_outputs2.append(models_outputs[4])
feature_importance2 = list()
feature_importance2.append(feature_importance[0])
feature_importance2.append(feature_importance[3])
feature_importance2.append(feature_importance[4])

fig, (ax1, ax2) = plt.subplots(2)
plt.rcParams["figure.figsize"] = [6.0, 4.0]
plt.rcParams["figure.autolayout"] = True
ax1.barh(models_outputs[1:3], feature_importance[1:3], 0.3 )
ax2.barh(models_outputs2,feature_importance2, 0.3)
ax1.title.set_text("EtOH Coefficients for All Outputs")
plt.xlabel('Raw Coefficient Values')
xabs_max = abs(max(ax1.get_xlim(), key=abs))
ax1.set_xlim(xmin=-xabs_max, xmax=xabs_max)
xabs_max2 = abs(max(ax2.get_xlim(), key=abs))
ax2.set_xlim(xmin=-xabs_max2, xmax=xabs_max2)
ax1.axvline(x=0, color=".5")
ax2.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
plt.show()







feature_importance = list()
feature_importance.append(density_average_feature_importance_element[3])
feature_importance.append(size_average_feature_importance_element[3])
feature_importance.append(BET_average_feature_importance_element[3])
feature_importance.append(micro_average_feature_importance_element[3])
feature_importance.append(total_average_feature_importance_element[3])

models_outputs2 = list()
models_outputs2.append(models_outputs[0])
models_outputs2.append(models_outputs[3])
models_outputs2.append(models_outputs[4])
feature_importance2 = list()
feature_importance2.append(feature_importance[0])
feature_importance2.append(feature_importance[3])
feature_importance2.append(feature_importance[4])

fig, (ax1, ax2) = plt.subplots(2)
plt.rcParams["figure.figsize"] = [6.0, 4.0]
plt.rcParams["figure.autolayout"] = True
ax1.barh(models_outputs[1:3], feature_importance[1:3], 0.3 )
ax2.barh(models_outputs2,feature_importance2, 0.3)
ax1.title.set_text("Water Coefficients for All Outputs")
plt.xlabel('Raw Coefficient Values')
xabs_max = abs(max(ax1.get_xlim(), key=abs))
ax1.set_xlim(xmin=-xabs_max, xmax=xabs_max)
xabs_max2 = abs(max(ax2.get_xlim(), key=abs))
ax2.set_xlim(xmin=-xabs_max2, xmax=xabs_max2)
ax1.axvline(x=0, color=".5")
ax2.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
plt.show()



feature_importance = list()
feature_importance.append(density_average_feature_importance_element[5])
feature_importance.append(size_average_feature_importance_element[5])
feature_importance.append(BET_average_feature_importance_element[5])
feature_importance.append(micro_average_feature_importance_element[5])
feature_importance.append(total_average_feature_importance_element[5])

models_outputs2 = list()
models_outputs2.append(models_outputs[0])
models_outputs2.append(models_outputs[3])
models_outputs2.append(models_outputs[4])
feature_importance2 = list()
feature_importance2.append(feature_importance[0])
feature_importance2.append(feature_importance[3])
feature_importance2.append(feature_importance[4])

fig, (ax1, ax2) = plt.subplots(2)
plt.rcParams["figure.figsize"] = [6.0, 4.0]
plt.rcParams["figure.autolayout"] = True
ax1.barh(models_outputs[1:3], feature_importance[1:3], 0.3 )
ax2.barh(models_outputs2,feature_importance2, 0.3)
ax1.title.set_text("Synthesis Time Coefficients for All Outputs")
plt.xlabel('Raw Coefficient Values')
xabs_max = abs(max(ax1.get_xlim(), key=abs))
ax1.set_xlim(xmin=-xabs_max, xmax=xabs_max)
xabs_max2 = abs(max(ax2.get_xlim(), key=abs))
ax2.set_xlim(xmin=-xabs_max2, xmax=xabs_max2)
ax1.axvline(x=0, color=".5")
ax2.axvline(x=0, color=".5")
plt.subplots_adjust(left=0.3)
plt.show()