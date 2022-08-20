# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:54:05 2022

@author: Khalid Alotaibi
"""
# Purpose: This file uses a random forest regressor to fit  quantitative results of MOF synthesis. It also checks the impact of converting these regression problems into classification. 
##################################################
# Step 1: Import needed libraries   
import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Step 2: Import Dataset
RandForDataset_all = pandas.read_csv("UTSA_16_A2ML_Data_Updated.csv")
RandForDataset = RandForDataset_all.dropna()     #drop all rows that have any NaN values

#print("The column headers :")
#print(list(RandForDataset.columns.values))
Y1 = np.array(RandForDataset['Bulk Density (g/ml)']);

condlist = ['Zinc Acetate Zn(CH3CO2)2 Mass (g)', 
            'Citric Acid HOC(CO2H)(CH2CO2H)2 Mass (g)', 
            'Potassium Hydroxide KOH Mass (g)', 'Water Volume (ml)', 
            'Ethanol EtOH Volume (ml)', 'Synthesis Time (hr)']

synthesis_features = ['Zn Mass', 
            'Acid Mass', 
            'Base Mass', 'Water Volume', 
            'EtOH Volume', 'Synthesis Time']

# Create the training and testing sets
training_data = RandForDataset.sample(frac=0.8, random_state=18)
testing_data = RandForDataset.drop(training_data.index)

X_training = np.array(training_data[condlist])
Y1_training = np.array(training_data['Bulk Density (g/ml)']);
Y2_training = np.array(training_data['Average Particle Size']);
Y3_training = np.array(training_data['BET Area (m2/g)'])
Y4_training = np.array(training_data['Micropore Volume (cm3/g)'])
Y6_training = np.array(training_data['Pore Volume (cm3/g)'])

X_testing = np.array(testing_data[condlist])
Y1_testing = np.array(testing_data['Bulk Density (g/ml)']);
Y2_testing = np.array(testing_data['Average Particle Size']);
Y3_testing = np.array(testing_data['BET Area (m2/g)'])
Y4_testing = np.array(testing_data['Micropore Volume (cm3/g)'])
Y6_testing = np.array(testing_data['Pore Volume (cm3/g)'])

# Define the number of variables/properties
number_of_variables = len(X_training[0])
number_of_training_datapoints = len(X_training)
number_of_testing_datapoints = len(X_testing)


# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
 
# create regressor object
regressor_density = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_size = RandomForestRegressor(n_estimators = 100, random_state=0) 
regressor_BET = RandomForestRegressor(n_estimators = 100, random_state=0) 
regressor_micropore = RandomForestRegressor(n_estimators = 100, random_state=0) 
regressor_mesopore = RandomForestRegressor(n_estimators = 100, random_state=0) 
regressor_tpore =RandomForestRegressor(n_estimators = 100, random_state=0) 

# fit the regressor with x and y data for Transparency 
regressor_density.fit(X_training, Y1_training) 
regressor_size.fit(X_training, Y2_training) 
regressor_BET.fit(X_training, Y3_training) 
regressor_micropore.fit(X_training, Y4_training) 
regressor_tpore.fit(X_training, Y6_training) 

# Estimate training data densities & particle sizes using the devleoped models
Y1_training_predicted = regressor_density.predict(X_training)
Y2_training_predicted = regressor_size.predict(X_training)
Y3_training_predicted = regressor_BET.predict(X_training)
Y4_training_predicted = regressor_micropore.predict(X_training)
Y6_training_predicted = regressor_tpore.predict(X_training)

# Predict transparency for testing set values
Y_pred_density = regressor_density.predict(X_testing)  # test the output by changing values
error_perc = (Y_pred_density-Y1_testing)/Y1_testing*100
print("Density",Y_pred_density,"",Y1_testing)
print(error_perc)

# Predict Average Particle Size for testing set values
Y_pred_size = regressor_size.predict(X_testing)  # test the output by changing values
error_perc = (Y_pred_size-Y2_testing)/Y2_testing*100
print("Particle Size",Y_pred_size,"",Y2_testing)
print(error_perc)

Y_pred_BET = regressor_BET.predict(X_testing)  # test the output by changing values
error_perc = (Y_pred_BET-Y3_testing)/Y3_testing*100
print("BET",Y_pred_BET,"",Y3_testing)
print(error_perc)

Y_pred_micropore = regressor_micropore.predict(X_testing)  # test the output by changing values
error_perc = (Y_pred_micropore-Y4_testing)/Y4_testing*100
print("Micropore Volume",Y_pred_micropore,"",Y4_testing)
print(error_perc)

Y_pred_tpore = regressor_tpore.predict(X_testing)  # test the output by changing values
error_perc = (Y_pred_tpore-Y6_testing)/Y6_testing*100
print("Total Volume",Y_pred_tpore,"",Y6_testing)
print(error_perc)

# Visualising the Random Forest Regression results for density using parity plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Actual Density (g/ml)', fontsize = 15)
ax.set_ylabel('Predicted Density (g/ml)', fontsize = 15)
ax.set_title('Parity Plot (RFR) - Bulk Density', fontsize = 20)
ax.scatter(Y1_training,Y1_training_predicted, s=50, marker="o")
ax.scatter(Y1_testing,Y_pred_density, s=50, marker="*")
ax.legend(['Training Data','Testing Data'],loc="upper right")
ax.grid()
## find the boundaries of X and Y values
bounds = (min(Y1_training.min(), Y1_training_predicted.min()) - (0.1 * Y1_training_predicted.min()), max(Y1_training.max(), Y1_training_predicted.max())+ (0.1 * Y1_training_predicted.max()))

# Reset the limits
ax = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
ax.set_aspect("equal", adjustable="box")
ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)
# Calculate Statistics of the Parity Plot 
mean_abs_err = np.mean(np.abs(Y1_training-Y1_training_predicted))
rmse = np.sqrt(np.mean((Y1_training-Y1_training_predicted)**2))
rmse_std = rmse / np.std(Y1_training_predicted)
z = np.polyfit(Y1_training,Y1_training_predicted, 1)
y_hat = np.poly1d(z)(Y1_training)

text = f"$\: \: Mean \: Absolute \: Error \: (MAE) = {mean_abs_err:0.3f}$ \n $ Root \: Mean \: Square \: Error \: (RMSE) = {rmse:0.3f}$ \n $ RMSE \: / \: Std(y) = {rmse_std :0.3f}$ \n $R^2 = {r2_score(Y1_training_predicted,y_hat):0.3f}$"

plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')


# Visualising the Random Forest Regression results for particle size using parity plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Actual Particle Size (nm)', fontsize = 15)
ax.set_ylabel('Predicted Particle Size (nm)', fontsize = 15)
ax.set_title('Parity Plot (RFR) - Particle Size', fontsize = 20)
ax.scatter(Y2_training,Y2_training_predicted, s=50, marker="o")
ax.scatter(Y2_testing,Y_pred_size, s=50,marker="*")
ax.legend(['Training Data','Testing Data'],loc="upper right")
ax.grid()
## find the boundaries of X and Y values
bounds = (min(Y2_training.min(), Y2_training_predicted.min()) - int(0.1 * Y2_training_predicted.min()), max(Y2_training.max(), Y2_training_predicted.max())+ int(0.1 * Y2_training_predicted.max()))

# Reset the limits
ax = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
ax.set_aspect("equal", adjustable="box")
ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)
# Calculate Statistics of the Parity Plot 
mean_abs_err = np.mean(np.abs(Y2_training-Y2_training_predicted))
rmse = np.sqrt(np.mean((Y2_training-Y2_training_predicted)**2))
rmse_std = rmse / np.std(Y2_training_predicted)
z = np.polyfit(Y2_training,Y2_training_predicted, 1)
y_hat = np.poly1d(z)(Y2_training)

text = f"$\: \: Mean \: Absolute \: Error \: (MAE) = {mean_abs_err:0.3f}$ \n $ Root \: Mean \: Square \: Error \: (RMSE) = {rmse:0.3f}$ \n $ RMSE \: / \: Std(y) = {rmse_std :0.3f}$ \n $R^2 = {r2_score(Y2_training_predicted,y_hat):0.3f}$"

plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')

# Visualising the Random Forest Regression results for BET using parity plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Actual BET Area ($m^2$/g)', fontsize = 15)
ax.set_ylabel('Predicted BET Area ($m^2$/g)', fontsize = 15)
ax.set_title('Parity Plot (RFR) - BET Area', fontsize = 20)
ax.scatter(Y3_training,Y3_training_predicted, s=50, marker="o")
ax.scatter(Y3_testing,Y_pred_BET, s=50,marker="*")
ax.legend(['Training Data','Testing Data'],loc="upper right")
ax.grid()
## find the boundaries of X and Y values
bounds = (min(Y3_training.min(), Y3_training_predicted.min()) - int(0.1 * Y3_training_predicted.min()), max(Y3_training.max(), Y3_training_predicted.max())+ int(0.1 * Y3_training_predicted.max()))

# Reset the limits
ax = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
ax.set_aspect("equal", adjustable="box")
ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)
# Calculate Statistics of the Parity Plot 
mean_abs_err = np.mean(np.abs(Y3_training-Y3_training_predicted))
rmse = np.sqrt(np.mean((Y3_training-Y3_training_predicted)**2))
rmse_std = rmse / np.std(Y3_training_predicted)
z = np.polyfit(Y3_training,Y3_training_predicted, 1)
y_hat = np.poly1d(z)(Y3_training)

text = f"$\: \: Mean \: Absolute \: Error \: (MAE) = {mean_abs_err:0.3f}$ \n $ Root \: Mean \: Square \: Error \: (RMSE) = {rmse:0.3f}$ \n $ RMSE \: / \: Std(y) = {rmse_std :0.3f}$ \n $R^2 = {r2_score(Y3_training_predicted,y_hat):0.3f}$"

plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')

# Visualising the Random Forest Regression results for Micropore volume using parity plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Actual Micropore Volume ($cm^3$/g)', fontsize = 15)
ax.set_ylabel('Predicted BET Area ($cm^3$/g)', fontsize = 15)
ax.set_title('Parity Plot (RFR) - Micropore Volume', fontsize = 20)
ax.scatter(Y4_training,Y4_training_predicted, s=50, marker="o")
ax.scatter(Y4_testing,Y_pred_micropore, s=50,marker="*")
ax.legend(['Training Data','Testing Data'],loc="upper right")
ax.grid()
## find the boundaries of X and Y values
bounds = (0.23, 0.3)

# Reset the limits
ax = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
ax.set_aspect("equal", adjustable="box")
ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)
# Calculate Statistics of the Parity Plot 
mean_abs_err = np.mean(np.abs(Y4_training-Y4_training_predicted))
rmse = np.sqrt(np.mean((Y4_training-Y4_training_predicted)**2))
rmse_std = rmse / np.std(Y4_training_predicted)
z = np.polyfit(Y4_training,Y4_training_predicted, 1)
y_hat = np.poly1d(z)(Y4_training)

text = f"$\: \: Mean \: Absolute \: Error \: (MAE) = {mean_abs_err:0.3f}$ \n $ Root \: Mean \: Square \: Error \: (RMSE) = {rmse:0.3f}$ \n $ RMSE \: / \: Std(y) = {rmse_std :0.3f}$ \n $R^2 = {r2_score(Y4_training_predicted,y_hat):0.3f}$"

plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')

# Visualising the Random Forest Regression results for Total pore volume using parity plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Actual Total Pore Volume ($cm^3$/g)', fontsize = 15)
ax.set_ylabel('Predicted Total Pore Volume ($cm^3$/g)', fontsize = 15)
ax.set_title('Parity Plot (RFR) - Total Pore Volume', fontsize = 20)
ax.scatter(Y6_training,Y6_training_predicted, s=50, marker="o")
ax.scatter(Y6_testing,Y_pred_tpore, s=50,marker="*")
ax.legend(['Training Data','Testing Data'],loc="upper right")
ax.grid()
## find the boundaries of X and Y values
bounds = (0.2, 0.6)

# Reset the limits
ax = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
ax.set_aspect("equal", adjustable="box")
ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)
# Calculate Statistics of the Parity Plot 
mean_abs_err = np.mean(np.abs(Y6_training-Y6_training_predicted))
rmse = np.sqrt(np.mean((Y6_training-Y6_training_predicted)**2))
rmse_std = rmse / np.std(Y6_training_predicted)
z = np.polyfit(Y6_training,Y6_training_predicted, 1)
y_hat = np.poly1d(z)(Y6_training)

text = f"$\: \: Mean \: Absolute \: Error \: (MAE) = {mean_abs_err:0.3f}$ \n $ Root \: Mean \: Square \: Error \: (RMSE) = {rmse:0.3f}$ \n $ RMSE \: / \: Std(y) = {rmse_std :0.3f}$ \n $R^2 = {r2_score(Y6_training_predicted,y_hat):0.3f}$"

plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')

# Converting the Density Regression problem into classification with intervals [1.5,1.66],[1.67,1.83], [1.83-2.00]:
intervals = np.array((1.50,1.66,1.88,2.00))
categories_in_numbers = np.array((100,101,102))
categories_in_ranges = ['1.50 - 1.66 g/ml','1.67 - 1.87 g/ml', '1.88 - 2.00 g/ml']

# Converting the Particle Size Regression problem into classification with intervals [0,100],[101,250], [251-inf]:
intervals_size = np.array((0,101,251,400))
categories_in_numbers_size = np.array((1000,1001,1002))
categories_in_ranges_size = ['0 - 100 nm','101 - 250 nm', '251 - 400 nm']
        
# Converting the BET Area Regression problem into classification with intervals [600,700],[701,800], [801-inf]:
intervals_BET = np.array((600,670,801,1000))
categories_in_numbers_BET = np.array((1000,1001,1002))
categories_in_ranges_BET = ['600 - 670 $m^2$/g','671 - 800 $m^2$/g', '801 < $m^2$/g']
    
# Converting the Micropore Volume Regression problem into classification:
intervals_micropore = np.array((0.23,0.25,0.27,0.3))
categories_in_numbers_micropore = np.array((1000,1001,1002))
categories_in_ranges_micropore = ['0.230 - 0.250 $cm^3$/g','0.251 - 0.270 $cm^3$/g', '0.270 < $cm^3$/g']

# Converting Y labels to categories described above
converted_Y1_training = Y1_training
converted_Y1_training = np.where(converted_Y1_training<intervals[1],categories_in_numbers[0],converted_Y1_training)
converted_Y1_training = np.where(converted_Y1_training<intervals[2],categories_in_numbers[1],converted_Y1_training)
converted_Y1_training = np.where(converted_Y1_training<intervals[3],categories_in_numbers[2],converted_Y1_training)

converted_Y1_testing = Y1_testing
converted_Y1_testing = np.where(converted_Y1_testing<intervals[1],categories_in_numbers[0],converted_Y1_testing)
converted_Y1_testing = np.where(converted_Y1_testing<intervals[2],categories_in_numbers[1],converted_Y1_testing)
converted_Y1_testing = np.where(converted_Y1_testing<intervals[3],categories_in_numbers[2],converted_Y1_testing)

converted_Y2_training = Y2_training
converted_Y2_training = np.where(converted_Y2_training<intervals_size[1],categories_in_numbers_size[0],converted_Y2_training)
converted_Y2_training = np.where(converted_Y2_training<intervals_size[2],categories_in_numbers_size[1],converted_Y2_training)
converted_Y2_training = np.where(converted_Y2_training<intervals_size[3],categories_in_numbers_size[2],converted_Y2_training)

converted_Y2_testing = Y2_testing
converted_Y2_testing = np.where(converted_Y2_testing<intervals_size[1],categories_in_numbers_size[0],converted_Y2_testing)
converted_Y2_testing = np.where(converted_Y2_testing<intervals_size[2],categories_in_numbers_size[1],converted_Y2_testing)
converted_Y2_testing = np.where(converted_Y2_testing<intervals_size[3],categories_in_numbers_size[2],converted_Y2_testing)

converted_Y3_training = Y3_training
converted_Y3_training = np.where(converted_Y3_training<intervals_BET[1],categories_in_numbers_BET[0],converted_Y3_training)
converted_Y3_training = np.where(converted_Y3_training<intervals_BET[2],categories_in_numbers_BET[1],converted_Y3_training)
converted_Y3_training = np.where(converted_Y3_training<intervals_BET[3],categories_in_numbers_BET[2],converted_Y3_training)

converted_Y3_testing = Y3_testing
converted_Y3_testing = np.where(converted_Y3_testing<intervals_BET[1],categories_in_numbers_BET[0],converted_Y3_testing)
converted_Y3_testing = np.where(converted_Y3_testing<intervals_BET[2],categories_in_numbers_BET[1],converted_Y3_testing)
converted_Y3_testing = np.where(converted_Y3_testing<intervals_BET[3],categories_in_numbers_BET[2],converted_Y3_testing)

converted_Y4_training = Y4_training
converted_Y4_training = np.where(converted_Y4_training<intervals_micropore[1],categories_in_numbers_micropore[0],converted_Y4_training)
converted_Y4_training = np.where(converted_Y4_training<intervals_micropore[2],categories_in_numbers_micropore[1],converted_Y4_training)
converted_Y4_training = np.where(converted_Y4_training<intervals_micropore[3],categories_in_numbers_micropore[2],converted_Y4_training)

converted_Y4_testing = Y4_testing
converted_Y4_testing = np.where(converted_Y4_testing<intervals_micropore[1],categories_in_numbers_micropore[0],converted_Y4_testing)
converted_Y4_testing = np.where(converted_Y4_testing<intervals_micropore[2],categories_in_numbers_micropore[1],converted_Y4_testing)
converted_Y4_testing = np.where(converted_Y4_testing<intervals_micropore[3],categories_in_numbers_micropore[2],converted_Y4_testing)



# performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_training_scaled = sc.fit_transform(X_training)
X_testing_scaled = sc.transform(X_testing)

# Random Forest Classifier Models
clf_1 = RandomForestClassifier()
clf_1.fit(X_training_scaled, converted_Y1_training)

clf_2_scaled = RandomForestClassifier()
clf_2_scaled.fit(X_training_scaled,converted_Y2_training)

clf_3 = RandomForestClassifier()
clf_3.fit(X_training_scaled, converted_Y3_training)

clf_4_scaled = RandomForestClassifier()
clf_4_scaled.fit(X_training_scaled,converted_Y4_training)

# Applying PCA function on training & testing sets 
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_all_components = PCA()

# X_values = np.array(RandForDataset[condlist])
# X_values = sc.fit_transform(X_values)

principal_components_training = pca.fit_transform(X_training_scaled,Y1_training)
principal_components_testing = pca.transform(X_testing_scaled)
principal_components_training_all = pca_all_components.fit_transform(X_training_scaled,Y1_training)
principal_components_testing_all = pca_all_components.transform(X_testing_scaled)

explained_variance_pca = pca.explained_variance_ratio_*100
explained_variance_pca_all_components = pca_all_components.explained_variance_ratio_*100

principalDf = pd.DataFrame(data = principal_components_training
              , columns = ['principal component 1', 'principal component 2'])

finalDf_density = pd.concat([principalDf, pd.DataFrame(Y1_training,columns=['Density 1 (g/ml)'])], axis = 1)
finalDf_density = pd.concat([finalDf_density, pd.DataFrame(converted_Y1_training,columns=['Density (g/ml)'])], axis = 1)
finalDf_density = pd.concat([finalDf_density, pd.DataFrame(X_training[:,0],columns=['Zn Mass (g)'])], axis = 1)
finalDf_density = pd.concat([finalDf_density, pd.DataFrame(X_training[:,4],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_density = pd.concat([finalDf_density, pd.DataFrame(X_training[:,3],columns=['Water Volume (ml)'])], axis = 1)

principal_components_training_size = pca.fit_transform(X_training_scaled,Y2_training)
principal_components_testing_size = pca.transform(X_testing_scaled)
principal_components_training_all_size = pca_all_components.fit_transform(X_training_scaled,Y2_training)
principal_components_testing_all_size = pca_all_components.transform(X_testing_scaled)

principalDf = pd.DataFrame(data = principal_components_training_size
              , columns = ['principal component 1', 'principal component 2'])

finalDf_size = pd.concat([principalDf, pd.DataFrame(Y2_training,columns=['Size 1'])], axis = 1)
finalDf_size = pd.concat([finalDf_size, pd.DataFrame(converted_Y2_training,columns=['Size'])], axis = 1)
finalDf_size = pd.concat([finalDf_size, pd.DataFrame(X_training[:,0],columns=['Zn Mass (g)'])], axis = 1)
finalDf_size = pd.concat([finalDf_size, pd.DataFrame(X_training[:,4],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_size = pd.concat([finalDf_size, pd.DataFrame(X_training[:,3],columns=['Water Volume (ml)'])], axis = 1)

finalDf_BET = pd.concat([principalDf, pd.DataFrame(Y3_training,columns=['BET 1'])], axis = 1)
finalDf_BET = pd.concat([finalDf_BET, pd.DataFrame(converted_Y3_training,columns=['BET'])], axis = 1)
finalDf_BET = pd.concat([finalDf_BET, pd.DataFrame(X_training[:,0],columns=['Zn Mass (g)'])], axis = 1)
finalDf_BET = pd.concat([finalDf_BET, pd.DataFrame(X_training[:,4],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_BET = pd.concat([finalDf_BET, pd.DataFrame(X_training[:,3],columns=['Water Volume (ml)'])], axis = 1)

finalDf_micropore = pd.concat([principalDf, pd.DataFrame(Y4_training,columns=['Micropore 1'])], axis = 1)
finalDf_micropore = pd.concat([finalDf_micropore, pd.DataFrame(converted_Y4_training,columns=['Micropore'])], axis = 1)
finalDf_micropore = pd.concat([finalDf_micropore, pd.DataFrame(X_training[:,0],columns=['Zn Mass (g)'])], axis = 1)
finalDf_micropore = pd.concat([finalDf_micropore, pd.DataFrame(X_training[:,4],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_micropore = pd.concat([finalDf_micropore, pd.DataFrame(X_training[:,3],columns=['Water Volume (ml)'])], axis = 1)

# Replace Density values with labels using For Loop
for i in range(len(Y1_training)):

 	if finalDf_density.iloc[i]['Density (g/ml)'] == 100:
         finalDf_density.loc[i,'Density (g/ml)'] = categories_in_ranges[0]
 	
 	if finalDf_density.iloc[i]['Density (g/ml)'] == 101:
         finalDf_density.loc[i,'Density (g/ml)'] = categories_in_ranges[1]
 	
 	if finalDf_density.iloc[i]['Density (g/ml)'] == 102:
         finalDf_density.loc[i,'Density (g/ml)'] = categories_in_ranges[2]
 	
# Replace size values with labels using For Loop
for i in range(len(Y2_training)):

 	if finalDf_size.iloc[i]['Size'] == 1000:
         finalDf_size.loc[i,'Size'] = categories_in_ranges_size[0]
    
 	if finalDf_size.iloc[i]['Size'] == 1001:
         finalDf_size.loc[i,'Size'] = categories_in_ranges_size[1]

 	if finalDf_size.iloc[i]['Size'] == 1002:
         finalDf_size.loc[i,'Size'] = categories_in_ranges_size[2]

# Replace size values with labels using For Loop
for i in range(len(Y3_training)):

 	if finalDf_BET.iloc[i]['BET'] == 1000:
         finalDf_BET.loc[i,'BET'] = categories_in_ranges_BET[0]
    
 	if finalDf_BET.iloc[i]['BET'] == 1001:
         finalDf_BET.loc[i,'BET'] = categories_in_ranges_BET[1]

 	if finalDf_BET.iloc[i]['BET'] == 1002:
         finalDf_BET.loc[i,'BET'] = categories_in_ranges_BET[2]
        
# Replace size values with labels using For Loop
for i in range(len(Y4_training)):

 	if finalDf_micropore.iloc[i]['Micropore'] == 1000:
         finalDf_micropore.loc[i,'Micropore'] = categories_in_ranges_micropore[0]
    
 	if finalDf_micropore.iloc[i]['Micropore'] == 1001:
         finalDf_micropore.loc[i,'Micropore'] = categories_in_ranges_micropore[1]

 	if finalDf_micropore.iloc[i]['Micropore'] == 1002:
         finalDf_micropore.loc[i,'Micropore'] = categories_in_ranges_micropore[2]
        
       
# Predicting the training set 
# result through scatter plot for Density using PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Density 2-Component PCA', fontsize = 20)
targets = categories_in_ranges
markers = ['o', 's','^']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_density['Density (g/ml)'] == target
    ax.scatter(finalDf_density.loc[indicesToKeep, 'principal component 1']
                , finalDf_density.loc[indicesToKeep, 'principal component 2']
                , marker = marker
                , s = 50)
ax.legend(targets)
ax.grid()

# Predicting the training set 
# result through scatter plot for Size using PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Size 2-Component PCA', fontsize = 20)
targets = categories_in_ranges_size
markers = ['v', '1','P']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_size['Size'] == target
    ax.scatter(finalDf_density.loc[indicesToKeep, 'principal component 1']
                , finalDf_density.loc[indicesToKeep, 'principal component 2']
                , marker = marker
                , s = 50)
ax.legend(targets)
ax.grid()

# Predicting the training set 
# result through scatter plot for BET using PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('BET Area 2-Component PCA', fontsize = 20)
targets = categories_in_ranges_BET
markers = ['o', '2','^']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_BET['BET'] == target
    ax.scatter(finalDf_density.loc[indicesToKeep, 'principal component 1']
                , finalDf_density.loc[indicesToKeep, 'principal component 2']
                , marker = marker
                , s = 50)
ax.legend(targets)
ax.grid()

# Predicting the training set 
# result through scatter plot for Micropore using PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Micropore Volume 2-Component PCA', fontsize = 20)
targets = categories_in_ranges_micropore
markers = ['s', 'v','P']
for target, marker in zip(targets,markers):
    indicesToKeep = finalDf_micropore['Micropore'] == target
    ax.scatter(finalDf_density.loc[indicesToKeep, 'principal component 1']
                , finalDf_density.loc[indicesToKeep, 'principal component 2']
                , marker = marker
                , s = 50)
ax.legend(targets)
ax.grid()

# Visualizing the explained variance represented by each PC
fig = plt.figure(figsize=(8,8))
plt.bar(['PC1','PC2','PC3','PC4'],explained_variance_pca_all_components)
ax = fig.add_subplot(1,1,1)
ax.set_title('Scree Plot of the Component Variances', fontsize=20)
ax.set_xlabel('Principal Component', fontsize=15)
ax.set_ylabel('Explained Variance (%)',fontsize=15)
ax.grid()
plt.grid(axis="x")
plt.show()


converted_Y1_training = np.where(converted_Y1_training==categories_in_numbers[0],0,converted_Y1_training)
converted_Y1_training = np.where(converted_Y1_training==categories_in_numbers[1],1,converted_Y1_training)
converted_Y1_training = np.where(converted_Y1_training==categories_in_numbers[2],2,converted_Y1_training)

converted_Y1_testing = np.where(converted_Y1_testing==categories_in_numbers[0],0,converted_Y1_testing)
converted_Y1_testing = np.where(converted_Y1_testing==categories_in_numbers[1],1,converted_Y1_testing)
converted_Y1_testing = np.where(converted_Y1_testing==categories_in_numbers[2],2,converted_Y1_testing)

principal_components_training = pca.fit_transform(X_training_scaled,Y1_training)
principal_components_testing = pca.transform(X_testing_scaled)
principal_components_training_all = pca_all_components.fit_transform(X_training_scaled,Y1_training)
principal_components_testing_all = pca_all_components.transform(X_testing_scaled)

explained_variance_pca = pca.explained_variance_ratio_*100
explained_variance_pca_all_components = pca_all_components.explained_variance_ratio_*100

principalDf = pd.DataFrame(data = principal_components_training
              , columns = ['principal component 1', 'principal component 2'])

finalDf_density = pd.concat([principalDf, pd.DataFrame(Y1_training,columns=['Density 1 (g/ml)'])], axis = 1)
finalDf_density = pd.concat([finalDf_density, pd.DataFrame(converted_Y1_training,columns=['Density (g/ml)'])], axis = 1)
finalDf_density = pd.concat([finalDf_density, pd.DataFrame(X_training[:,0],columns=['Zn Mass (g)'])], axis = 1)
finalDf_density = pd.concat([finalDf_density, pd.DataFrame(X_training[:,4],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_density = pd.concat([finalDf_density, pd.DataFrame(X_training[:,3],columns=['Water Volume (ml)'])], axis = 1)


converted_Y2_training = np.where(converted_Y2_training==categories_in_numbers_size[0],0,converted_Y2_training)
converted_Y2_training = np.where(converted_Y2_training==categories_in_numbers_size[1],1,converted_Y2_training)
converted_Y2_training = np.where(converted_Y2_training==categories_in_numbers_size[2],2,converted_Y2_training)

converted_Y2_testing = np.where(converted_Y2_testing==categories_in_numbers_size[0],0,converted_Y2_testing)
converted_Y2_testing = np.where(converted_Y2_testing==categories_in_numbers_size[1],1,converted_Y2_testing)
converted_Y2_testing = np.where(converted_Y2_testing==categories_in_numbers_size[2],2,converted_Y2_testing)

principal_components_training_size = pca.fit_transform(X_training_scaled,Y2_training)
principal_components_testing_size = pca.transform(X_testing_scaled)
principal_components_training_all_size = pca_all_components.fit_transform(X_training_scaled,Y2_training)
principal_components_testing_all_size = pca_all_components.transform(X_testing_scaled)

explained_variance_pca_all_components_size = pca_all_components.explained_variance_ratio_*100

principalDf = pd.DataFrame(data = principal_components_training_size
              , columns = ['principal component 1', 'principal component 2'])

finalDf_size = pd.concat([principalDf, pd.DataFrame(Y2_training,columns=['Size 1'])], axis = 1)
finalDf_size = pd.concat([finalDf_size, pd.DataFrame(converted_Y2_training,columns=['Size'])], axis = 1)
finalDf_size = pd.concat([finalDf_size, pd.DataFrame(X_training[:,0],columns=['Zn Mass (g)'])], axis = 1)
finalDf_size = pd.concat([finalDf_size, pd.DataFrame(X_training[:,4],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_size = pd.concat([finalDf_size, pd.DataFrame(X_training[:,3],columns=['Water Volume (ml)'])], axis = 1)


converted_Y3_training = np.where(converted_Y3_training==categories_in_numbers_BET[0],0,converted_Y3_training)
converted_Y3_training = np.where(converted_Y3_training==categories_in_numbers_BET[1],1,converted_Y3_training)
converted_Y3_training = np.where(converted_Y3_training==categories_in_numbers_BET[2],2,converted_Y3_training)
converted_Y3_testing = np.where(converted_Y3_testing==categories_in_numbers_BET[0],0,converted_Y3_testing)
converted_Y3_testing = np.where(converted_Y3_testing==categories_in_numbers_BET[1],1,converted_Y3_testing)
converted_Y3_testing = np.where(converted_Y3_testing==categories_in_numbers_BET[2],2,converted_Y3_testing)

finalDf_BET = pd.concat([principalDf, pd.DataFrame(Y3_training,columns=['BET 1'])], axis = 1)
finalDf_BET = pd.concat([finalDf_BET, pd.DataFrame(converted_Y3_training,columns=['BET'])], axis = 1)
finalDf_BET = pd.concat([finalDf_BET, pd.DataFrame(X_training[:,0],columns=['Zn Mass (g)'])], axis = 1)
finalDf_BET = pd.concat([finalDf_BET, pd.DataFrame(X_training[:,4],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_BET = pd.concat([finalDf_BET, pd.DataFrame(X_training[:,3],columns=['Water Volume (ml)'])], axis = 1)


converted_Y4_training = np.where(converted_Y4_training==categories_in_numbers_micropore[0],0,converted_Y4_training)
converted_Y4_training = np.where(converted_Y4_training==categories_in_numbers_micropore[1],1,converted_Y4_training)
converted_Y4_training = np.where(converted_Y4_training==categories_in_numbers_micropore[2],2,converted_Y4_training)
converted_Y4_testing = np.where(converted_Y4_testing==categories_in_numbers_micropore[0],0,converted_Y4_testing)
converted_Y4_testing = np.where(converted_Y4_testing==categories_in_numbers_micropore[1],1,converted_Y4_testing)
converted_Y4_testing = np.where(converted_Y4_testing==categories_in_numbers_micropore[2],2,converted_Y4_testing)

finalDf_micropore = pd.concat([principalDf, pd.DataFrame(Y4_training,columns=['Micropore 1'])], axis = 1)
finalDf_micropore = pd.concat([finalDf_micropore, pd.DataFrame(converted_Y4_training,columns=['Micropore'])], axis = 1)
finalDf_micropore = pd.concat([finalDf_micropore, pd.DataFrame(X_training[:,0],columns=['Zn Mass (g)'])], axis = 1)
finalDf_micropore = pd.concat([finalDf_micropore, pd.DataFrame(X_training[:,4],columns=['EtOH Volume (ml)'])], axis = 1)
finalDf_micropore = pd.concat([finalDf_micropore, pd.DataFrame(X_training[:,3],columns=['Water Volume (ml)'])], axis = 1)

# # Visualizing the weight/loading of each feature for PC1 & PC2 & Transparency Values

# colormap = ['r','g','b','y','k','c']

# def myplot(score,coeff,labels=None):
#     xs = score[:,0]
#     ys = score[:,1]
#     n = coeff.shape[0]

#     plt.scatter(xs ,ys, c = Y1_training) #without scaling
#     fig = plt.figure(figsize = (8,8))
#     ax = fig.add_subplot(1,1,1) 
#     ax.set_xlabel('PC 1', fontsize = 15)
#     ax.set_ylabel('PC 2', fontsize = 15)
#     ax.set_title('PCA Biplot', fontsize = 20)
#     targets = categories_in_ranges
#     colors = ['g', 'r','b']
#     for target, color in zip(targets,colors):
#         indicesToKeep = finalDf_density['Density (g/ml)'] == target
#         ax.scatter(finalDf_density.loc[indicesToKeep, 'principal component 1']
#                    , finalDf_density.loc[indicesToKeep, 'principal component 2']
#                    , c = color
#                    , s = 50)
#     ax.legend(targets)
#     ax.grid()
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = colormap[i],alpha = 0.5)
#         if labels is None:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, synthesis_features[i], color = colormap[i], ha = 'center', va = 'center', fontsize=8)
#         else:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

# #Call the function. 
# myplot(principal_components_training_all[:,0:2], pca_all_components. components_) #principal_components_training with pca & principal_component_training_all with pca_all_components
# plt.show()

# def myplot(score,coeff,labels=None):
#     xs = score[:,0]
#     ys = score[:,1]
#     n = coeff.shape[0]
#     plt.xlim(-1,1)
#     plt.ylim(-1,1)
#     plt.title("Scores of Features on PC1 & PC2")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.grid()

#     for i in range(n):
#         plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = colormap[i],alpha = 0.5)
#         if labels is None:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, synthesis_features[i], color = colormap[i], ha = 'center', va = 'center', fontsize=8)
#         else:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')


# #Call the function. 
# myplot(principal_components_training_all[:,0:2], pca_all_components. components_) #principal_components_training with pca & principal_component_training_all with pca_all_components
# plt.show()

# Redoing classification based on scaled data
clf_1_scaled = RandomForestClassifier()
clf_1_scaled.fit(principal_components_training,converted_Y1_training)

clf_2_scaled = RandomForestClassifier()
clf_2_scaled.fit(principal_components_training_size,converted_Y2_training)

clf_3_scaled = RandomForestClassifier()
clf_3_scaled.fit(principal_components_training_size, converted_Y3_training)

clf_4_scaled = RandomForestClassifier()
clf_4_scaled.fit(principal_components_training_size,converted_Y4_training)

# Fitting Logistic Regression To the training set
from sklearn.linear_model import LogisticRegression 
 
classifier_density = LogisticRegression(random_state = 0)
classifier_density.fit(principal_components_training, converted_Y1_training)

classifier_size = LogisticRegression(random_state = 0)
classifier_size.fit(principal_components_training_size, converted_Y2_training)

classifier_BET = LogisticRegression(random_state = 0)
classifier_BET.fit(principal_components_training_size, converted_Y3_training)

classifier_micropore = LogisticRegression(random_state = 0)
classifier_micropore.fit(principal_components_training_size, converted_Y4_training)

# Predicting the training set 
# Density results through scatter plot for using PCA data and a random forest model followed by logistic regression
converted_Y1_training = converted_Y1_training.astype('int')

from matplotlib.colors import ListedColormap

X_set, y2_set = principal_components_training_all, converted_Y1_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_1_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = categories_in_ranges
categories = [0,1,2]

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['o', 's', '^'][i], label = legend_2[j])

plt.title('Density Random Forest Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


converted_Y1_testing = converted_Y1_testing.astype('int')

X_set, y1_set, ss = principal_components_testing_all, converted_Y1_testing, principal_components_training_all

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_1_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = categories_in_ranges

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['o', 's', '^'][i], label = legend_1[j])

plt.title('Density Random Forest Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()

X_set, y2_set = principal_components_training_all, converted_Y1_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_density.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = categories_in_ranges

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['o', 's', '^'][i], label = legend_2[j])

plt.title('Density Logistic Regrission Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


converted_Y1_testing = converted_Y1_testing.astype('int')

X_set, y1_set, ss = principal_components_testing_all, converted_Y1_testing, principal_components_training_all

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_density.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = categories_in_ranges

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['o', 's', '^'][i], label = legend_1[j])

plt.title('Density Logistic Regression Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()


# Predicting the training set 
# Size results through scatter plot for using PCA data and a random forest model followed by logistic regression
converted_Y2_training = converted_Y2_training.astype('int')

from matplotlib.colors import ListedColormap

X_set, y2_set = principal_components_training_all_size, converted_Y2_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_2_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = categories_in_ranges_size

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['v', '1', 'P'][i], label = legend_2[j])

plt.title('Particle Size Random Forest Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


converted_Y2_testing = converted_Y2_testing.astype('int')

X_set, y1_set, ss = principal_components_testing_all_size, converted_Y2_testing, principal_components_training_all_size

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_2_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = categories_in_ranges_size

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['v', '1', 'P'][i], label = legend_1[j])

plt.title('Particle Size Random Forest Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()

converted_Y2_training = converted_Y2_training.astype('int')

from matplotlib.colors import ListedColormap

X_set, y2_set = principal_components_training_all_size, converted_Y2_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_size.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = categories_in_ranges_size

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['v', '1', 'P'][i], label = legend_2[j])

plt.title('Particle Size Logistic Regression Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


converted_Y2_testing = converted_Y2_testing.astype('int')

X_set, y1_set, ss = principal_components_testing_all_size, converted_Y2_testing, principal_components_training_all_size

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_size.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = categories_in_ranges_size

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['v', '1', 'P'][i], label = legend_1[j])

plt.title('Particle Size Logistic Regression Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()


# BET figures
converted_Y3_training = converted_Y3_training.astype('int')

from matplotlib.colors import ListedColormap

X_set, y2_set = principal_components_training_all_size, converted_Y3_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_3_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = categories_in_ranges_BET

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['o', '2','^'][i], label = legend_2[j])

plt.title('BET Area Random Forest Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


converted_Y3_testing = converted_Y3_testing.astype('int')

X_set, y1_set, ss = principal_components_testing_all_size, converted_Y3_testing, principal_components_training_all_size

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_3_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = categories_in_ranges_BET

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['o', '2','^'][i], label = legend_1[j])

plt.title('BET Area Random Forest Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()

from matplotlib.colors import ListedColormap

X_set, y2_set = principal_components_training_all_size, converted_Y3_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_BET.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = categories_in_ranges_BET

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['o', '2','^'][i], label = legend_2[j])

plt.title('BET Area Logistic Regression Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


X_set, y1_set, ss = principal_components_testing_all_size, converted_Y3_testing, principal_components_training_all_size

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_BET.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = categories_in_ranges_BET

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['o', '2','^'][i], label = legend_1[j])

plt.title('BET Area Logistic Regression Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()





# Micropore Volume figures
converted_Y4_training = converted_Y4_training.astype('int')

from matplotlib.colors import ListedColormap

X_set, y2_set = principal_components_training_all_size, converted_Y4_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_4_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = categories_in_ranges_micropore

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['s', 'v','P'][i], label = legend_2[j])

plt.title('Micropore Volume Random Forest Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


converted_Y4_testing = converted_Y4_testing.astype('int')

X_set, y1_set, ss = principal_components_testing_all_size, converted_Y4_testing, principal_components_training_all_size

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, clf_4_scaled.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = categories_in_ranges_micropore

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['s', 'v','P'][i], label = legend_1[j])

plt.title('Micropore Volume Random Forest Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()

from matplotlib.colors import ListedColormap

X_set, y2_set = principal_components_training_all_size, converted_Y4_training

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
 					stop = X_set[:, 0].max() + 1, step = 0.01),
 					np.arange(start = X_set[:, 1].min() - 1,
 					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_micropore.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_2 = categories_in_ranges_micropore

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y2_set == j, 0], X_set[y2_set == j, 1],
				marker = ['s', 'v','P'][i], label = legend_2[j])

plt.title('Micropore Volume Logistic Regression Classification (Training set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()


X_set, y1_set, ss = principal_components_testing_all_size, converted_Y4_testing, principal_components_training_all_size

X1, X2 = np.meshgrid(np.arange(start = ss[:, 0].min() - 1,
 					stop = ss[:, 0].max() + 1, step = 0.01),
 					np.arange(start = ss[:, 1].min() - 1,
 					stop = ss[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier_micropore.predict(np.array([X1.ravel(),
 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
 			cmap = ListedColormap(('#326da8','#a88932', '#32a863')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

legend_1 = categories_in_ranges_micropore

for i, j in enumerate(np.unique(categories)):
 	plt.scatter(X_set[y1_set == j, 0], X_set[y1_set == j, 1],
				marker = ['s', 'v','P'][i], label = legend_1[j])

plt.title('Micropore Volume Logistic Regression Classification (Testing set) - PCA')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend


# show scatter plot
plt.show()









# # Combining density & particle size to obtain a new classification
# Y3_training = [0]*number_of_training_datapoints
# categories_combined = ['Low Density & Small Particle Size',
#                        'Low Density & Medium Particle Size',
#                        'Low Density & Large Particle Size',
#                        'Medium Density & Small Particle Size',
#                        'Medium Density & Medium Particle Size',
#                        'Medium Density & Large Particle Size',
#                        'High Density & Small Particle Size',
#                        'High Density & Medium Particle Size',
#                        'High Density & Large Particle Size']

# for i in range(number_of_training_datapoints):
#     if converted_Y1_training[i] == 0 and converted_Y2_training[i] == 0:
#         Y3_training[i] = 'Low Density & Small Particle Size'
#     elif converted_Y1_training[i] == 0 and converted_Y2_training[i] == 1:
#         Y3_training[i] = 'Low Density & Medium Particle Size'
#     elif converted_Y1_training[i] == 0 and converted_Y2_training[i] == 2:
#         Y3_training[i] = 'Low Density & Large Particle Size'
#     elif converted_Y1_training[i] == 1 and converted_Y2_training[i] == 0:
#         Y3_training[i] = 'Medium Density & Small Particle Size'
#     elif converted_Y1_training[i] == 1 and converted_Y2_training[i] == 1:
#         Y3_training[i] = 'Medium Density & Medium Particle Size'
#     elif converted_Y1_training[i] == 1 and converted_Y2_training[i] == 2:
#         Y3_training[i] = 'Medium Density & Large Particle Size'
#     elif converted_Y1_training[i] == 2 and converted_Y2_training[i] == 0:
#         Y3_training[i] = 'High Density & Small Particle Size'
#     elif converted_Y1_training[i] == 2 and converted_Y2_training[i] == 1:
#         Y3_training[i] = 'High Density & Medium Particle Size'
#     elif converted_Y1_training[i] == 2 and converted_Y2_training[i] == 2:
#         Y3_training[i] = 'High Density & Large Particle Size'
        
# Y3_testing = [0]*number_of_testing_datapoints
       
# for i in range(number_of_testing_datapoints):
#     if converted_Y1_testing[i] == 0 and converted_Y2_testing[i] == 0:
#         Y3_testing[i] = 'Low Density & Small Particle Size'
#     elif converted_Y1_testing[i] == 0 and converted_Y2_testing[i] == 1:
#         Y3_testing[i] = 'Low Density & Medium Particle Size'
#     elif converted_Y1_testing[i] == 0 and converted_Y2_testing[i] == 2:
#         Y3_testing[i] = 'Low Density & Large Particle Size'
#     elif converted_Y1_testing[i] == 1 and converted_Y2_testing[i] == 0:
#         Y3_testing[i] = 'Medium Density & Small Particle Size'
#     elif converted_Y1_testing[i] == 1 and converted_Y2_testing[i] == 1:
#         Y3_testing[i] = 'Medium Density & Medium Particle Size'
#     elif converted_Y1_testing[i] == 1 and converted_Y2_testing[i] == 2:
#         Y3_testing[i] = 'Medium Density & Large Particle Size'
#     elif converted_Y1_testing[i] == 2 and converted_Y2_testing[i] == 0:
#         Y3_testing[i] = 'High Density & Small Particle Size'
#     elif converted_Y1_testing[i] == 2 and converted_Y2_testing[i] == 1:
#         Y3_testing[i] = 'High Density & Medium Particle Size'
#     elif converted_Y1_testing[i] == 2 and converted_Y2_testing[i] == 2:
#         Y3_testing[i] = 'High Density & Large Particle Size'
        
# converted_Y3_training = [0]*number_of_training_datapoints

# for i in range(number_of_training_datapoints):
#     if Y3_training[i] == 'Low Density & Small Particle Size':
#         converted_Y3_training[i] = 0
#     elif Y3_training[i] == 'Low Density & Medium Particle Size':
#         converted_Y3_training[i] = 1
#     elif Y3_training[i] == 'Low Density & Large Particle Size':
#         converted_Y3_training[i] = 2    
#     elif Y3_training[i] == 'Medium Density & Small Particle Size':
#         converted_Y3_training[i] = 3        
#     elif Y3_training[i] == 'Medium Density & Medium Particle Size':
#         converted_Y3_training[i] = 4
#     elif Y3_training[i] == 'Medium Density & Large Particle Size':
#         converted_Y3_training[i] = 5        
#     elif Y3_training[i] == 'High Density & Small Particle Size':
#         converted_Y3_training[i] = 6
#     elif Y3_training[i] == 'High Density & Medium Particle Size':
#         converted_Y3_training[i] = 7
#     elif Y3_training[i] == 'High Density & Large Particle Size':
#         converted_Y3_training[i] = 8

# converted_Y3_testing = [0]*number_of_testing_datapoints              
        
# for i in range(number_of_testing_datapoints):
#     if Y3_testing[i] == 'Low Density & Small Particle Size':
#         converted_Y3_testing[i] = 0
#     elif Y3_testing[i] == 'Low Density & Medium Particle Size':
#         converted_Y3_testing[i] = 1
#     elif Y3_testing[i] == 'Low Density & Large Particle Size':
#         converted_Y3_testing[i] = 2    
#     elif Y3_testing[i] == 'Medium Density & Small Particle Size':
#         converted_Y3_testing[i] = 3        
#     elif Y3_testing[i] == 'Medium Density & Medium Particle Size':
#         converted_Y3_testing[i] = 4
#     elif Y3_testing[i] == 'Medium Density & Large Particle Size':
#         converted_Y3_testing[i] = 5        
#     elif Y3_testing[i] == 'High Density & Small Particle Size':
#         converted_Y3_testing[i] = 6
#     elif Y3_testing[i] == 'High Density & Medium Particle Size':
#         converted_Y3_testing[i] = 7
#     elif Y3_testing[i] == 'High Density & Large Particle Size':
#         converted_Y3_testing[i] = 8
