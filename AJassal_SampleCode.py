# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:10:57 2017

@author: akash
"""
'''
Question: How Much Electricity Are You Using?
Using Statistical Models to Predict Your Energy Consumption.

'''
'''
from tkinter import *

#WALLTYPE','ROOFTYPE','YEARMADE','AIA_Zone','BEDROOMS','ADQINSUL
class Application(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.grid()
        self.create_form()
        
    def create_form(self):
        self.text_desc = Label(self, wraplength=400, text = "Select the options to define the house to predict energy consumption.  All fields are required")
        self.text_desc.grid(row=0, column=0, columnspan=2, sticky = NW)
        
        self.walltype_text = Label(self, text = "Wall Type:")
        self.walltype_text.grid(row=2, column=0, columnspan=1, rowspan=9, sticky = NW)

        self.walltype_list = Listbox(self, selectmode=SINGLE, height=9,exportselection=0)
        self.walltype_list.grid(row=3, column=1, rowspan=9, sticky=NW)
        self.walltype_list.insert(1,"Brick")   
        self.walltype_list.insert(2,"Wood")   
        self.walltype_list.insert(3,"Siding")
        self.walltype_list.insert(4,"Stucco")   
        self.walltype_list.insert(5,"Composition")   
        self.walltype_list.insert(6,"Stone")
        self.walltype_list.insert(7,"Concrete")   
        self.walltype_list.insert(8,"Glass")   
        self.walltype_list.insert(9,"Other")  
        
        self.rooftype_text = Label(self, text = "Roof Type:")
        self.rooftype_text.grid(row=12, column=0, columnspan=1, sticky = NW)
        
        self.rooftype_list = Listbox(self, selectmode=SINGLE, height=8,exportselection=0)
        self.rooftype_list.grid(row=12, column=1, rowspan=8, sticky=NW)
        self.rooftype_list.insert(1,"Ceramic/Clay Tile")   
        self.rooftype_list.insert(2,"Wood Shingles")   
        self.rooftype_list.insert(3,"Metal")
        self.rooftype_list.insert(4,"Slate")   
        self.rooftype_list.insert(5,"Composition")   
        self.rooftype_list.insert(6,"Asphalt")
        self.rooftype_list.insert(7,"Concrete")   
        self.rooftype_list.insert(8,"Other") 
        
        self.yearmade_text = Label(self, text = "Year House constructed:")
        self.yearmade_text.grid(row =20, column=0, columnspan=1, sticky = NW)
        self.yearmade = Entry(self)
        self.yearmade.grid(row=20, column=1, sticky=NW)
        
        self.aia_zone_text = Label(self, text = "AIA Climate Zone:")
        self.aia_zone_text.grid(row=21, column=0, columnspan=1, sticky = NW)
        self.aia_zone = Entry(self)
        self.aia_zone.grid(row=21, column=1, sticky=NW)
        
        self.bedrooms_text = Label(self, text = "Number of Bedrooms:")
        self.bedrooms_text.grid(row=22, column =0, columnspan=1, sticky = NW)
        self.bedrooms = Entry(self)
        self.bedrooms.grid(row=22, column=1, sticky=NW)
        
        self.adqinsul_text = Label(self, text = "Adequate Insulation:")
        self.adqinsul_text.grid(row=23, column=0, columnspan=1, sticky = NW)
        
        self.adqinsul_list = Listbox(self, selectmode=SINGLE, height=4,exportselection=0)
        self.adqinsul_list.grid(row=23, column=1, rowspan=4, sticky=NW)
        self.adqinsul_list.insert(1,"Well Insulated")   
        self.adqinsul_list.insert(2,"Adequately Insulated")   
        self.adqinsul_list.insert(3,"Poorly Insulated")
        self.adqinsul_list.insert(4,"No Insulation")   
        
        self.submit_button = Button(self, text="Submit", command=self.process)
        self.submit_button.grid(row=28, column=0, sticky=W)
        
        self.text = Text(self, width=35, height=5, wrap = WORD)
        self.text.grid(row=30, column=0, columnspan=2, sticky=NW)
        
    def process(self):
        walltype = sum(self.walltype_list.curselection(), 1)
        rooftype = sum(self.rooftype_list.curselection(), 1)
        yearmade = self.yearmade.get()
        aia_zone = self.aia_zone.get()
        bedrooms = self.bedrooms.get()
        adqinsul = sum(self.adqinsul_list.curselection(), 1)
        ### Prediction -- KWH_Range
        X_new = np.array([[walltype, rooftype, yearmade, aia_zone, bedrooms, adqinsul]])
        prediction = knn.predict(X_new)
        prediction_text = "Prediction: %d KWH Range Mean" % prediction
        #print('Prediction:', prediction,'KWH Range Mean')
        #prediction_text = "Walltype: %s" % walltype
        self.text.delete(0.0, END)
        self.text.insert(0.0, prediction_text)
'''
########################################
# ENVIRONMENT PREP
import os
### Provide the path here
os.chdir('C:\\Users\\akash\\Desktop\\GWU\\6103_DataMining_FBradley\\Project_1') 

### Basic Packages
import pandas as pd

## DataLoad and Global Filtering

### read in the 2009 survey data from CSV 
complete_energy_df  = pd.read_csv('recs2009_public2.csv')
keys_df = pd.read_csv('public_layout.csv')

### Create More Managable DataFrame
energy_df = complete_energy_df[['KWH','WALLTYPE','ROOFTYPE','YEARMADE','REGIONC','BEDROOMS','ADQINSUL','TYPEGLASS','WINDOWS','DOOR1SUM','FUELHEAT','TOTSQFT','AUDIT','NUMPC','NHSLDMEM','NUMFRIG','TVCOLOR','TOTROOMS']]

del(complete_energy_df)
del(keys_df)
########################################
# DATA TIDYING

## Null Management 

### Traditional Null Check
energy_df.dropna(inplace=True)

### Null Check
feature_colsA = list(energy_df)
nulls_df = pd.DataFrame()
i=0
while i < len(feature_colsA):
    print(feature_colsA[i])
    print((energy_df[feature_colsA[i]] ==-2).sum())
    nulls_df[feature_colsA[i]] = (energy_df[feature_colsA[i]] ==-2).sum()
    i+=1

#### Null Correction
#remove all records where nulls exist (value = -2)
#energy_df = energy_df[(energy_df >= 0).all(1)]

########################################
#del(nulls_df,i, feature_colsA, energy_df)
########################################

# Create X and Y Dataframes

x_energy_df = energy_df.drop('KWH',axis=1,inplace=False)
x_feature_cols = list(x_energy_df)

########################################
# BINNING Y-Variable
import numpy as np

# create KWH_range values to group the energy usage into ranges
conditions = [
     (energy_df['KWH'] >= 0) & (energy_df['KWH'] < 1000),
     (energy_df['KWH'] >= 2000) & (energy_df['KWH'] < 2500),
     (energy_df['KWH'] >= 2500) & (energy_df['KWH'] < 3000),
     (energy_df['KWH'] >= 3000) & (energy_df['KWH'] < 3500),
     (energy_df['KWH'] >= 3500) & (energy_df['KWH'] < 4500),
     (energy_df['KWH'] >= 4500) & (energy_df['KWH'] < 5000),
     (energy_df['KWH'] >= 5000) & (energy_df['KWH'] < 5500),
     (energy_df['KWH'] >= 5500) & (energy_df['KWH'] < 6000),
     (energy_df['KWH'] >= 6000) & (energy_df['KWH'] < 6500),
     (energy_df['KWH'] >= 6500) & (energy_df['KWH'] < 7000),
     (energy_df['KWH'] >= 7000) & (energy_df['KWH'] < 7500),
     (energy_df['KWH'] >= 7500) & (energy_df['KWH'] < 8000),
     (energy_df['KWH'] >= 8000) & (energy_df['KWH'] < 8500),
     (energy_df['KWH'] >= 8500) & (energy_df['KWH'] < 9000),
     (energy_df['KWH'] >= 9000) & (energy_df['KWH'] < 9500),
     (energy_df['KWH'] >= 9500) & (energy_df['KWH'] < 10000),
     (energy_df['KWH'] >= 10000) & (energy_df['KWH'] < 10500),
     (energy_df['KWH'] >= 10500) & (energy_df['KWH'] < 11000),
     (energy_df['KWH'] >= 11000) & (energy_df['KWH'] < 11500),
     (energy_df['KWH'] >= 11500) & (energy_df['KWH'] < 12000),
     (energy_df['KWH'] >= 12000) & (energy_df['KWH'] < 12500),
     (energy_df['KWH'] >= 12500) & (energy_df['KWH'] < 13000),
     (energy_df['KWH'] >= 13000) & (energy_df['KWH'] < 13500),
     (energy_df['KWH'] >= 13500) & (energy_df['KWH'] < 14500),
     (energy_df['KWH'] >= 14500) & (energy_df['KWH'] < 15000),
     (energy_df['KWH'] >= 15000) & (energy_df['KWH'] < 15500),
     (energy_df['KWH'] >= 15500) & (energy_df['KWH'] < 16000),
     (energy_df['KWH'] >= 16000) & (energy_df['KWH'] < 16500),
     (energy_df['KWH'] >= 16500) & (energy_df['KWH'] < 17000),
     (energy_df['KWH'] >= 17000) & (energy_df['KWH'] < 17500),
     (energy_df['KWH'] >= 17500) & (energy_df['KWH'] < 18000),
     (energy_df['KWH'] >= 18000) & (energy_df['KWH'] < 18500),
     (energy_df['KWH'] >= 18500) & (energy_df['KWH'] < 19000),
     (energy_df['KWH'] >= 19000) & (energy_df['KWH'] < 19500),
     (energy_df['KWH'] >= 19500) & (energy_df['KWH'] < 20000),
     (energy_df['KWH'] >= 20000) & (energy_df['KWH'] < 25000),
      (energy_df['KWH'] >= 25000) & (energy_df['KWH'] < 50000),
      (energy_df['KWH'] >= 50000) & (energy_df['KWH'] < 75000),
      (energy_df['KWH'] >= 75000)]
choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
energy_df['KWH_range'] = np.select(conditions, choices, default=38)
y2_energy_df = energy_df[['KWH_range']]
'''
data = energy_df[['KWH']].copy()
kwh_min = min(energy_df['KWH'])
kwh_max = max(energy_df['KWH'])

kwh_bins = [np.linspace(kwh_min, kwh_max, 1000)]
kwh_bins = np.ravel(kwh_bins)
digitized = np.digitize(data, kwh_bins)
kwh_means = [data[digitized == i].KWH.mean() for i in range(1,len(kwh_bins))]
kwh_max   = [data[digitized == i].KWH.max()  for i in range(1,len(kwh_bins))]
kwh_min   = [data[digitized == i].KWH.min()  for i in range(1,len(kwh_bins))]

#result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])
#Convert Bins from List to DataFrames
kwh_means_df = pd.DataFrame(kwh_means, columns = ['kwh_range_mean'])
kwh_min_df   = pd.DataFrame(kwh_min, columns   = ['kwh_range_min'])
kwh_max_df   = pd.DataFrame(kwh_max, columns   = ['kwh_range_max'])

kwh_bins     = pd.concat([kwh_means_df, kwh_min_df, kwh_max_df], axis=1)
#kwh_bins = kwh_bins.fillna(9e8).astype(int)

kwh_bins.dropna(inplace=True)
kwh_bins = kwh_bins[np.isfinite(kwh_bins['kwh_range_mean'])]

#Match back to Orginal Dataset
data['kwh_binned'] = pd.cut(data['KWH'], kwh_bins['kwh_range_min'])
data['kwh_binned'] = pd.IntervalIndex(data['kwh_binned'],dtype='interval[int64]').left
#data['kwh_binned'] = data['kwh_binned'].astype(int)

#labels = ['kwh_range_mean']
#kwh_bins = pd.DataFrame(kwh_means, columns = labels)
#delete this ^^

#kwh_bins = kwh_bins[np.isfinite(kwh_bins['kwh_range_mean'])]
#target_energy_df = kwh_bins[np.isfinite(kwh_bins['kwh_range_mean'])]
target_energy_df = data[['kwh_binned']].copy()
target_energy_df['kwh_binned'] = target_energy_df['kwh_binned'].dropna().astype(int)
target_energy_df[['kwh_binned']] = target_energy_df[['kwh_binned']].apply(pd.to_numeric) 
'''
#y_energy_df = energy_df[['KWH']]

#y2_energy_df = energy_df[['KWH']]
########################################
#del(energy_df,x_energy_df, x_feature_cols, y2_energy_df)
########################################
#MODELING (DECISION TREE)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
########################################
# PARTITION DATA

## Split data into training set & test set for Decision Tree
X_train, X_test, Y_train, Y_test =tts(x_energy_df, y2_energy_df, test_size = 0.3, random_state=6103)
########################################
# TREE DEPTH
## list of values to try
max_depth_range = range(1, 12)
### list to store the average MSE for each value of max_depth
all_MSE_scores = []

## calculate MSE score for each value of max_depth
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=6103)
    MSE_scores = cross_val_score(treereg, X_train, Y_train, cv=14, scoring='neg_mean_squared_error')
    all_MSE_scores.append(np.mean(np.sqrt(-MSE_scores)))
    
## plot max_depth (x-axis) versus MSE (y-axis)
plt.plot(max_depth_range, all_MSE_scores)
plt.title('Max Depth Range Plot (Decision Tree)')
plt.xlabel('max_depth')
plt.ylabel('MSE (lower is better)')
plt.savefig('decisiontree1.png')
plt.show()

# Depth of 7
########################################
# FEATURE IMPORTANCE

treereg = DecisionTreeRegressor(max_depth=7, random_state=6103)
treereg.fit(X_train, Y_train)
# max_depth7 was best, so fit a tree using that parameter

## "Gini importance" of each feature: 
print(pd.DataFrame({'feature':x_feature_cols, 'importance':sorted(treereg.feature_importances_ *1000, reverse = True)}))

del(all_MSE_scores,MSE_scores,depth)
########################################
# New DATASET (Based on Decision Tree Model)

x2_energy_df = x_energy_df[['WALLTYPE','ROOFTYPE','YEARMADE','REGIONC','BEDROOMS','ADQINSUL']].copy()


## Feature_Cols Based on Results from Decision Tree Model
feature_colsB = list(x2_energy_df)

## split data into training set & test set
X_train, X_test, Y_train, Y_test =tts(x2_energy_df, y2_energy_df, test_size = 0.5, random_state=6103)
########################################
del()
########################################
# EDA
from pandas.plotting import scatter_matrix

##DV {KWH} Summary

plt.hist(energy_df.KWH_range, bins = 39)
#plt.axis([0,30000,0,160])
plt.grid = True
plt.savefig('hist1.png')
plt.show()
#plt.hist(energy_df.KWH_range_means, bins=1000)
#plt.savefig('hist_2.png')
'''
plt.figure()
plt.boxplot(energy_df.kwh_binned, 0, 'gD')
plt.savefig('boxplot1.png')
plt.show()

## make a Scatterplot.1
scatter_matrix(x2_energy_df, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.savefig('scatter_matrix1.png')
plt.show()
'''
#########################################################
#MODELING (KNN)
from sklearn.neighbors import KNeighborsClassifier

## knn tuning --> What K should we pick
### capture results
train_accuracy = []
test_accuracy  = []
### set kNN setting from 1 to 11
kNN_range = range(1, 11)
for neighbors in kNN_range:
  # start Nearest Neighbors Classifier with K of 1
  knn = KNeighborsClassifier(n_neighbors=neighbors,
                             metric='minkowski', p=2)
  # train the data using Nearest Neighbors
  knn.fit(X_train, Y_train)
  # capture training accuracy
  train_accuracy.append(knn.score(X_train, Y_train))
  # predict using the test dataset  
  Y_pred = knn.predict(X_test)
  # capture test accuracy
  test_accuracy.append(knn.score(X_test, Y_test))
### plot results  
plt.plot(kNN_range, train_accuracy, label='training accuracy')
plt.plot(kNN_range, test_accuracy,  label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Neighbors')
plt.legend()
plt.savefig('knn_nums1.png')
plt.show()

## start Nearest Neighbors Classifier with K of 7
knn = KNeighborsClassifier(n_neighbors=7,
                metric='minkowski', p=2)

### train the data using Nearest Neighbors
knn.fit(X_train, Y_train)

### Model accuracy
Y_pred= knn.predict(X_test)
print('\nPrediction from X_test:')
print(Y_pred)

score = knn.score(X_test, Y_test)
print('Score:', score*100,'%')

### Prediction -- KWH_Range
X_new = np.array([[3, 5,1992,3,3,1]])
prediction = knn.predict(X_new)
print('Prediction:', prediction,'KWH Range')

'''
### Prediction -- KWH

## split data into training set & test set
target_energy_df = energy_df[['KWH']]
X_train, X_test, Y_train, Y_test =tts(data_energy_dfB, target_energy_df, test_size = 0.3, random_state=6103)

## start Nearest Neighbors Classifier with K of 7
knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
### train the data using Nearest Neighbors
knn.fit(X_train, Y_train)

X_new = np.array([[1, 5,1995,3,5,1]])
prediction = knn.predict(X_new)
print('Prediction:', prediction,'KWH')
'''
#########################################################
'''
root = Tk()

root.title("simple gui")
root.geometry("400x610")
app = Application(root)
root.mainloop()
'''
########################################################
########################################################
# 'WALLTYPE','ROOFTYPE','YEARMADE','REGIONC','BEDROOMS','ADQINSUL'

'''
(1)  Walltype
1.Brick
2.Wood
3.Siding (Aluminum, Vinyl, Steel)
4.Stucco
5.Composition (Shingle)
6.Stone
7.Concrete/Concrete Block
8.Glass
9. Other"
==
(2) Rooftype
1.Ceramic or Clay Tiles
2.Wood Shingles/Shakes
3.Metal
4.Slate or Synthetic Slate
5.Composition Shingles
6.Asphalt
7.Concrete Tiles
8.Other
9.Not Applicable
==
(3) Yearmade
1600 - 2009
==
(4) Region 
1.Northeast Census Region
2.Midwest Census Region
3.South Census Region
4.West Census Region
==
(5) Bedrooms
-2, 0:20
==
(6) ADQINSUL
1.Well Insulated
2.Adequately Insulated
3.Poorly Insulated
4.No Insulation
==
'''
'''
##A-a-ron's Random Forest
from sklearn.ensemble import RandomForestClassifier
labels=data_energy_df.columns[:]
XX = data_energy_df.iloc[:, 1:].values
YY = data_energy_df.iloc[:, 0].values
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(XX,YY)
'''

## Do we need one-hot encoding --> dont think we have to
## Do we need to impute or remove nulls (-2) --> done
## Do we need extra variables (ie: sqft) --> yes, need to do 
## Create 10x10 Scatter plot with x-axis = KWH 
## Front-end design if possible 
## Figure out Accuracy of Data (use IRIS Slide 14/24) --> done
## Binning process for KWH --> automate for n-bins
## Remove Outliers maybe --> dont think we have to