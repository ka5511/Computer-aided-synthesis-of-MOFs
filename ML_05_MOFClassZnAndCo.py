# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:53:36 2022

@author: Khalid Alotaibi
"""
# Purpose: Perform supervised classification modeling for MOFs shape and quality using Random Forest and Logistic Regression algorithms and plotting results in 2-d PC space
####################################################################

# Step 1: Import needed libraries   
import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
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

# # Random Forest Classifier Models Using Actual Features
# clf_1 = RandomForestClassifier()
# clf_1.fit(X_training, Y1_training)

# clf_2 = RandomForestClassifier()
# clf_2.fit(X_training, Y2_training)

# clf_3 = RandomForestClassifier()
# clf_3.fit(X_training,Y3_training)

# # Feature Importance
# print("Feature importance for determining transparency and hardness")
# print("")
# for x in range(number_of_variables):
#     print(condlist[x]) 
#     print("Transparency:", clf_1.feature_importances_[x])
#     print("Hardness:", clf_2.feature_importances_[x])
#     print("Both:", clf_3.feature_importances_[x])
#     print("")

# # Predictions of testing set
# predict_transparency = clf_1.predict(X_testing)
# predict_hardness = clf_2.predict(X_testing)
# predict_both = clf_3.predict(X_testing)
# print(predict_transparency)
# print(Y1_testing)
# print(predict_hardness)
# print(Y2_testing)
# print(predict_both)
# print(Y3_testing)
# print("")

# for x in range(number_of_testing_datapoints):
#     print(predict_transparency[x])
#     if predict_transparency[x] == Y1_testing[x]:
#         print("Prediction is Correct")
#     else:
#         print("Prediction is Incorrect")
#     print("")
#     print(predict_hardness[x])
#     if predict_hardness[x] == Y2_testing[x]:
#         print("Prediction is Correct")
#     else:
#         print("Prediction is Incorrect")
#     print("")

# # User input data
# user_X_testing = []
# Zn_mass = float(input("Please enter the mass of zinc acetate used in (g): "))
# user_X_testing.append(Zn_mass)
# CA_mass = float(input("Please enter the mass of citric acid used in (g): "))
# user_X_testing.append(CA_mass)
# KOH_mass= float(input("Please enter the mass of KOH used in (g): "))
# user_X_testing.append(KOH_mass)
# water_volume = float(input("Please enter the volume of water used in (ml): "))
# user_X_testing.append(water_volume)
# ethanol_volume = float(input("Please enter the volume of EtOH used in (ml): "))
# user_X_testing.append(ethanol_volume)
# syn_time = float(input("Please enter the synthesis time in (hr): "))
# user_X_testing.append(syn_time)
# user_X_testing = np.array(user_X_testing)
# user_X_testing.resize(1,6)
# print("Predicted Monolith Transparency: ", clf_1.predict(user_X_testing))
# print("Predicted Hardness: ", clf_2.predict(user_X_testing))

# performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_training_scaled = sc.fit_transform(X_training)
X_testing_scaled = sc.transform(X_testing)

# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

principal_components_training = pca.fit_transform(X_training_scaled,Y1_training)
principal_components_training_hardness = pca.fit_transform(X_training_scaled,Y2_training)

principal_components_testing = pca.transform(X_testing_scaled)
principal_components_testing_hardness = pca.transform(X_testing_scaled)


explained_variance_pca = pca.explained_variance_ratio_

# Redoing classification based on scaled data
clf_1_scaled = RandomForestClassifier(criterion="entropy",random_state=0)
clf_1_scaled.fit(principal_components_training,Y1_training)


clf_2_scaled = RandomForestClassifier(criterion="entropy",random_state=0)
clf_2_scaled.fit(principal_components_training_hardness,Y2_training)

clf_3_scaled = RandomForestClassifier(criterion="entropy",random_state=0)
clf_3_scaled.fit(principal_components_training,Y3_training)

# Fitting Logistic Regression To the training set
from sklearn.linear_model import LogisticRegression 
 
classifier_transparency = LogisticRegression(random_state = 0)
classifier_transparency.fit(principal_components_training, Y1_training)

classifier_hardness = LogisticRegression(random_state = 0)
classifier_hardness.fit(principal_components_training_hardness, Y2_training)

classifier_both = LogisticRegression(random_state = 0)
classifier_both.fit(principal_components_training, Y3_training)

# Predicting the test set result using
# predict function under RandomForest
predict_transparency = clf_1_scaled.predict(principal_components_testing)
predict_hardness = clf_2_scaled.predict(principal_components_testing)
predict_both = clf_3_scaled.predict(principal_components_testing)

# predict function under LogisticRegression
y_pred_transparncy = classifier_transparency.predict(principal_components_testing)
y_pred_hardness = classifier_hardness.predict(principal_components_testing)
y_pred_both = classifier_both.predict(principal_components_testing)

# making confusion matrix between
# test set of Y1 & Y2 and their predicted values.
from sklearn.metrics import confusion_matrix

cm1_RF = confusion_matrix(Y1_testing, predict_transparency)
cm1_LR = confusion_matrix(Y1_testing, y_pred_transparncy)
cm2_RF = confusion_matrix(Y2_testing, predict_hardness)
cm2_LR = confusion_matrix(Y2_testing, y_pred_hardness)
cm3_RF = confusion_matrix(Y3_testing, predict_both)
cm3_LR = confusion_matrix(Y3_testing, y_pred_both)

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

# Redoing classification based on scaled data for both RF and LR
clf_1_scaled = RandomForestClassifier(criterion="entropy",random_state=0,class_weight='balanced')
clf_1_scaled.fit(principal_components_training,Y1_training)

clf_2_scaled = RandomForestClassifier(criterion="entropy",random_state=0, class_weight='balanced')
clf_2_scaled.fit(principal_components_training_hardness,Y2_training)

clf_3_scaled = RandomForestClassifier(criterion="entropy",random_state=0,class_weight='balanced')
clf_3_scaled.fit(principal_components_training,Y3_training)

classifier_transparency = LogisticRegression(random_state = 0,class_weight='balanced')
classifier_transparency.fit(principal_components_training, Y1_training)

classifier_hardness = LogisticRegression(random_state = 0,class_weight='balanced')
classifier_hardness.fit(principal_components_training_hardness, Y2_training)

classifier_both = LogisticRegression(random_state = 0,class_weight='balanced')
classifier_both.fit(principal_components_training, Y3_training)

# Predicting the training set 
# Transparency results through scatter plot using PCA data and a random forest model
from matplotlib.colors import ListedColormap

# X_set, y1_set = principal_components_training, Y1_training

# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
#  					stop = X_set[:, 0].max() + 1, step = 0.01),
#  					np.arange(start = X_set[:, 1].min() - 1,
#  					stop = X_set[:, 1].max() + 1, step = 0.01))

# plt.contourf(X1, X2, clf_1_scaled.predict(np.array([X1.ravel(),
#  			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
#  			cmap = ListedColormap(('#326da8','#a88932')))

# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())

# legend_1 = ["Transparent","White-colored"]

# for i, j in enumerate(np.unique(y1_set)):
#  	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
# 				marker = ['o','^'][i], label = legend_1[j])

# plt.title('Transparency Random Forest Classification (Training set) - PCA')
# plt.xlabel('PC1') # for Xlabel
# plt.ylabel('PC2') # for Ylabel
# plt.legend() # to show legend


# # show scatter plot
# plt.show()


# X_set, y1_set, ss = principal_components_testing, Y1_testing, principal_components_training

# X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
#  					stop = ss[:, 0].max() + 1, step = 0.01),
#  					np.arange(start = ss[:, 1].min() - 1,
#  					stop = ss[:, 1].max() + 1, step = 0.01))

# plt.contourf(X1, X2, clf_1_scaled.predict(np.array([X1.ravel(),
#  			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
#  			cmap = ListedColormap(('#326da8','#a88932')))

# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())

# legend_1 = ["Transparent","White-colored"]

# for i, j in enumerate(np.unique(y1_set)):
#  	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
# 				marker = ['o','^'][i], label = legend_1[j])

# plt.title('Transparency Random Forest Classification (Testing set) - PCA')
# plt.xlabel('PC1') # for Xlabel
# plt.ylabel('PC2') # for Ylabel
# plt.legend() # to show legend


# # show scatter plot
# plt.show()


# # Predicting the training set 
# # Transparency results through scatter plot for using PCA data and a logistic regrission model 
# X_set, y1_set = principal_components_training, Y1_training

# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
#  					stop = X_set[:, 0].max() + 1, step = 0.01),
#  					np.arange(start = X_set[:, 1].min() - 1,
#  					stop = X_set[:, 1].max() + 1, step = 0.01))

# plt.contourf(X1, X2, classifier_transparency.predict(np.array([X1.ravel(),
#  			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
#  			cmap = ListedColormap(('#326da8','#a88932')))

# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())

# legend_1 = ["Transparent","White-colored"]

# for i, j in enumerate(np.unique(y1_set)):
#  	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
# 				marker = ['o','^'][i], label = legend_1[j])

# plt.title('Transparency Logistic Regression Classification (Training set) - PCA')
# plt.xlabel('PC1') # for Xlabel
# plt.ylabel('PC2') # for Ylabel
# plt.legend() # to show legend


# # show scatter plot
# plt.show()

# X_set, y1_set, ss = principal_components_testing, Y1_testing, principal_components_training

# X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
#  					stop = ss[:, 0].max() + 1, step = 0.01),
#  					np.arange(start = ss[:, 1].min() - 1,
#  					stop = ss[:, 1].max() + 1, step = 0.01))

# plt.contourf(X1, X2, classifier_transparency.predict(np.array([X1.ravel(),
#  			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
#  			cmap = ListedColormap(('#326da8','#a88932')))

# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())

# legend_1 = ["Transparent","White-colored"]

# for i, j in enumerate(np.unique(y1_set)):
#  	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
# 				marker = ['o','^'][i], label = legend_1[j])

# plt.title('Transparency Logistic Regression Classification (Testing set) - PCA')
# plt.xlabel('PC1') # for Xlabel
# plt.ylabel('PC2') # for Ylabel
# plt.legend() # to show legend


# # show scatter plot
# plt.show()


# Predicting the training set 
# Hardness results through scatter plot for using PCA data and a random forest model
X_set, y2_set = principal_components_training_hardness, Y2_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_2_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = ["Monolith","Powder"]

for i, j in enumerate(np.unique(y2_set)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['s','*'][i], label = legend_2[j])

plt.title('MOF Shape Random Forest Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()



X_set, y1_set, ss = principal_components_testing_hardness, Y2_testing, principal_components_training_hardness

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_2_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = ["Monolith","Powder"]

for i, j in enumerate(np.unique(y1_set)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['s','*'][i], label = legend_1[j])

plt.title('MOF Shape Random Forest Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()



# Predicting the training set 
# Hardness results through scatter plot for using PCA data and a Logistic Regression model
X_set, y2_set = principal_components_training_hardness, Y2_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_hardness.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = ["Monolith","Powder"]

for i, j in enumerate(np.unique(y2_set)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['s','*'][i], label = legend_2[j])

plt.title('MOF Shape Logistic Regression Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


X_set, y1_set, ss = principal_components_testing_hardness, Y2_testing, principal_components_training_hardness

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_hardness.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = ["Monolith","Powder"]

for i, j in enumerate(np.unique(y1_set)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['s','*'][i], label = legend_1[j])

plt.title('MOF Shape Logistic Regression Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()


# Predicting the training set 
# 'Combined' results through scatter plot using PCA data and a random forest model
X_set, y2_set = principal_components_training, Y3_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_3_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = ["Transparent Monolith","White-colored Monolith","Powder"]

for i, j in enumerate(np.unique(y2_set)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['h','+','1'][i], label = legend_2[j])

plt.title('MOF Quality Random Forest Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()



X_set, y1_set, ss = principal_components_testing, Y3_testing, principal_components_training

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_3_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = ["Transparent Monolith","White-colored Monolith","Powder"]

for i, j in enumerate(np.unique(y1_set)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['h','+','1'][i], label = legend_1[j])

plt.title('MOF Quality Random Forest Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()

# Predicting the training set 
# Hardness results through scatter plot for using PCA data and a Logistic Regression model
X_set, y2_set = principal_components_training, Y3_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_both.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = ["Transparent Monolith","White-colored Monolith","Powder"]

for i, j in enumerate(np.unique(y2_set)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['h','+','1'][i], label = legend_2[j])

plt.title('MOF Quality Logistic Regression Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


X_set, y1_set, ss = principal_components_testing, Y3_testing, principal_components_training

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_both.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = ["Transparent Monolith","White-colored Monolith","Powder"]

for i, j in enumerate(np.unique(y1_set)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['h','+','1'][i], label = legend_1[j])

plt.title('MOF Quality Logistic Regression Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()
