# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:28:39 2022

@author: Khalid Alotaibi
"""
# Purpose: Program the first version of the MONOSYN Prediction Tool

####################################################################
#                       MONOSYN Prediction Tool                    #
####################################################################

# Step 1: Import libraries   
import numpy as np
import joblib
from tabulate import tabulate

##################### Import best models ################################
# Data scaling
filename2 = "Data_Scaling.joblib"
sc = joblib.load(filename2)
# PCA 
filename2 = "PCA.joblib"
pca = joblib.load(filename2)
# Shape Coding Map
filename2 = "Shape_Codes.joblib"
shape_codes = joblib.load(filename2)
shape_codes["Powder"] = shape_codes.pop("N")
shape_codes["Monolith"] = shape_codes.pop("Y")
# Quality Coding Map
filename2 = "Quality_Codes.joblib"
quality_codes = joblib.load(filename2)
# MOF Shape Model
filename2 = "Shape_Model.joblib"
shape_model = joblib.load(filename2)
# MOF Quality Model
filename2 = "Quality_Model.joblib"
quality_model = joblib.load(filename2)
# MOF Density Model
filename2 = "Density_Model.joblib"
density_model = joblib.load(filename2)
# MOF Size Model
filename2 = "Size_Model.joblib"
size_model = joblib.load(filename2)
# MOF BET Model
filename2 = "BET_Model.joblib"
BET_model = joblib.load(filename2)
# MOF Micropore Volume Model
filename2 = "Micro_Model.joblib"
micro_model = joblib.load(filename2)
# MOF Total Pore Volume Model
filename2 = "Total_Volume_Model.joblib"
total_volume_model = joblib.load(filename2)



#####################    User Interaction ################################
user_X_testing = []
user_X_testing_reg = []
print("                                                                                 WELCOME TO THE MOF PREDICTION TOOL (MOFPT)\nThis tool predicts the quality, bulk density, average particle size, BET area, micropore volume, mesopore volume and total pore volume of a UTSA-16 Zn-based MOF & predicts the quality of UTSA-16 Co-based MOF based on seven user-specified synthesis conditions.\nOnce the synthesis conditions are specified, a full description of the synthesis procedure will be displayed for user's review and confirmation.")
print("")
Co = 1
Zn = 2
report_reg = "No"
mistakes = 0 # could you used to add different messages based on the number of times the user makes mistakes

while True:
    while True:
        metal_type = float(input("Please specify UTSA-16 metal source\n For cobalt acetate, please enter 1\n For zinc acetate, please enter 2\n Selected metal source number is: "))
        if metal_type == Co or metal_type == Zn:
            break
        else:
            print("\n Please choose a valid metal source!")
          
    while True:
        if metal_type == 1:
            Co_mass = float(input("Please enter the mass of cobalt acetate in (g): "))
            if Co_mass>0:
                user_X_testing.append(Co_mass)
                user_X_testing.append(0)
                break
            else:
                print("Hopefully this was a typo, please be careful!")
                
        else:
            Zn_mass = float(input("Please enter the mass of zinc acetate in (g): "))
            if Zn_mass>0:
                user_X_testing.append(0)
                user_X_testing.append(Zn_mass)
                user_X_testing_reg.append(Zn_mass)
                report_reg = "Yes"
                break
            else:
                print("Hopefully this was a typo, please be careful!")
    
    while True:
        CA_mass = float(input("Please enter the mass of citric acid in (g): "))
        if CA_mass>0:
            user_X_testing.append(CA_mass)
            user_X_testing_reg.append(CA_mass)
            break
        else:
            print("Hopefully this was a typo, please be careful!")
        
    while True:
        KOH_mass= float(input("Please enter the mass of KOH used in (g): "))
        if KOH_mass>0:
            user_X_testing.append(KOH_mass)
            user_X_testing_reg.append(KOH_mass)
            break
        else:
            print("Hopefully this was a typo, please be careful!")
            
    while True:
        water_volume = float(input("Please enter the volume of water used in (ml): "))
        if water_volume>0:
            user_X_testing.append(water_volume)
            user_X_testing_reg.append(water_volume)
            break
        else:
            print("Hopefully this was a typo, please be careful!")
    
    while True:
        ethanol_volume = float(input("Please enter the volume of EtOH used in (ml): "))
        if ethanol_volume>=0:
            user_X_testing.append(ethanol_volume)
            user_X_testing_reg.append(ethanol_volume)
            break
        else:
            print("Hopefully this was a typo, please be careful!")
    
    while True:
        syn_time = float(input("Please enter the synthesis time in (hr): "))
        if syn_time>0:
            user_X_testing.append(syn_time)
            user_X_testing_reg.append(syn_time)
            break
        else:
            print("Hopefully this was a typo, please be careful!")
    print("\nThe proposed experiment procedure is shown below. If the experiment procedure is correct, please enter 'Yes' to simulate the results. Otherwise, please enter 'No'")
    if report_reg=="Yes":
        decision = str(input("A total of %0.2f g of zinc acetate dihydrate is added to a solution of %0.2f g of anhydrous citric acid in %0.1f ml of water and the solution is stirred for 20 minutes to dissolve solids. A total of %0.2f g of potassium hydroxide is added to the solution and the solution is stirred for 10 minutes to dissolve solids. After that, %0.1f ml of ethanol is added and the solution is stirred for 30 minutes. The obtained solution is inserted into a thermal oven at 100 C for %0.1f hours. The sample is washed and centrifuged twice with water for 30 minutes each. Finally, the sample is left to dry at room temperature.\n Is this procedure correct: " %(Zn_mass, CA_mass, water_volume, KOH_mass, ethanol_volume, syn_time)))
        if decision == "YES" or decision == "Yes" or decision == "yes" or decision == "YEs" or decision == "yES" or decision == "yeS":
            break
    elif report_reg == "No":
        decision = str(input("A total of %0.2f g of cobalt acetate tetrahydrate is added to a solution of %0.2f g of anhydrous citric acid in %0.1f ml of water and the solution is stirred for 20 minutes to dissolve solids. A total of %0.2f g of potassium hydroxide is added to the solution and the solution is stirred for 10 minutes to dissolve solids. After that, %0.1f ml of ethanol is added and the solution is stirred for 30 minutes. The obtained solution is inserted into a thermal oven at 100 C for %0.1f hours. The sample is washed and centrifuged twice with water for 30 minutes each. Finally, the sample is left to dry at room temperature.\n Is this procedure correct: " %(Co_mass, CA_mass, water_volume, KOH_mass, ethanol_volume, syn_time)))
        if decision == "YES" or decision == "Yes" or decision == "yes" or decision == "YEs" or decision == "yES" or decision == "yeS":
            break

user_X_testing = np.array(user_X_testing)
user_X_testing.resize(1,7)

user_X_testing_reg = np.array(user_X_testing_reg)
user_X_testing_reg.resize(1,6)
# Prediction of MOF Shape & Quality
X_scaled = sc.transform(user_X_testing)
principal_components=pca.transform(X_scaled)
predicted_shape = shape_model.predict(principal_components)
predicted_quality = quality_model.predict(principal_components)
quality = list(quality_codes.keys())[list(quality_codes.values()).index(predicted_quality)]
# print(quality)

# Prediction of Density, Particle Size, BET, Micro, and Total Volume
print("")
if report_reg == "Yes":
    if quality != "Powder":
        density = density_model.predict(user_X_testing_reg)
        # print("Estimated Bulk Density is %0.3f g/cm\u00b3" %density)
        size = size_model.predict(user_X_testing_reg)
        # print("Estimated Average Particle Size is %0.3f nm" %size)
        BET = BET_model.predict(user_X_testing_reg)
        # print("Estimated BET Area is %0.3f m\u00b2/g" %BET)
        micro = micro_model.predict(user_X_testing_reg)
        # print("Estimated Micropore Volume is %0.3f cm\u00b3/g" %micro)
        total_volume = total_volume_model.predict(user_X_testing_reg)
        # print("Estimated Total Pore Volume is %0.3f cm\u00b3/g" %total_volume)
        meso = total_volume - micro
        # Print all predictions
        if density > 0 and size > 0 and BET > 0 and micro > 0 and meso > 0:
            data = [[quality, np.around(density,3), np.around(size,0), np.around(BET,0), np.around(micro,3), np.around(meso,3), np.around(total_volume,3)]]
            print (tabulate(data, headers=["MOF Quality", "Density (g/cm\u00b3)", "Average Particle Size (nm)", "BET Area (m\u00b2/g)","Micropore Volume (cm\u00b3/g)","Mesopore Volume (cm\u00b3/g)","Total Pore Volume (cm\u00b3/g)"]))
        else:
            print("Prediction results are unreliable. It is recommended to conduct the experiment in a lab and feed the results back to the tool to improve it.")
    else:
        data = [[quality, "Prediction is unreliable", "Prediction is unreliable", "Prediction is unreliable", "Prediction is unreliable", "Prediction is unreliable", "Prediction is unreliable"]]
        print (tabulate(data, headers=["MOF Quality", "Density (g/cm\u00b3)", "Average Particle Size (nm)", "BET Area (m\u00b2/g)","Micropore Volume (cm\u00b3/g)","Mesopore Volume (cm\u00b3/g)","Total Pore Volume (cm\u00b3/g)"]))

            
else:
    data = [[quality]]
    print (tabulate(data, headers=["MOF Quality"]))
