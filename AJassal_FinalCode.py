# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:10:57 2017

@author: SIJ_Energy
"""
'''
Question: How Much Energy Will You Use?
Using Statistical Models to Predict Your Energy Consumption.
'''

from tkinter import *

#WALLTYPE','ROOFTYPE','YEARMADE','REGIONC','BEDROOMS','ADQINSUL
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
        self.walltype_list.insert(1,"1. Brick")   
        self.walltype_list.insert(2,"2. Wood")   
        self.walltype_list.insert(3,"3. Siding")
        self.walltype_list.insert(4,"4. Stucco")   
        self.walltype_list.insert(5,"5. Composition")   
        self.walltype_list.insert(6,"6. Stone")
        self.walltype_list.insert(7,"7. Concrete")   
        self.walltype_list.insert(8,"8. Glass")   
        self.walltype_list.insert(9,"9. Other")  
        
        self.rooftype_text = Label(self, text = "Roof Type:")
        self.rooftype_text.grid(row=12, column=0, columnspan=1, sticky = NW)
        
        self.rooftype_list = Listbox(self, selectmode=SINGLE, height=8,exportselection=0)
        self.rooftype_list.grid(row=12, column=1, rowspan=8, sticky=NW)
        self.rooftype_list.insert(1,"1. Ceramic/Clay Tile")   
        self.rooftype_list.insert(2,"2. Wood Shingles")   
        self.rooftype_list.insert(3,"3. Metal")
        self.rooftype_list.insert(4,"4. Slate")   
        self.rooftype_list.insert(5,"5. Composition")   
        self.rooftype_list.insert(6,"6. Asphalt")
        self.rooftype_list.insert(7,"7. Concrete")   
        self.rooftype_list.insert(8,"8. Other") 
        
        self.yearmade_text = Label(self, text = "Year House constructed:")
        self.yearmade_text.grid(row =20, column=0, columnspan=1, sticky = NW)
        self.yearmade = Entry(self)
        self.yearmade.grid(row=20, column=1, sticky=NW)
        
        self.region_text = Label(self, text = "Region:")
        self.region_text.grid(row=21, column=0, columnspan=1, sticky = NW)
        self.region = Entry(self)
        self.region.grid(row=21, column=1, sticky=NW)
        
        self.bedrooms_text = Label(self, text = "Number of Bedrooms:")
        self.bedrooms_text.grid(row=22, column =0, columnspan=1, sticky = NW)
        self.bedrooms = Entry(self)
        self.bedrooms.grid(row=22, column=1, sticky=NW)
        
        self.adqinsul_text = Label(self, text = "Adequate Insulation:")
        self.adqinsul_text.grid(row=23, column=0, columnspan=1, sticky = NW)
        
        self.adqinsul_list = Listbox(self, selectmode=SINGLE, height=4,exportselection=0)
        self.adqinsul_list.grid(row=23, column=1, rowspan=4, sticky=NW)
        self.adqinsul_list.insert(1,"1. Well Insulated")   
        self.adqinsul_list.insert(2,"2. Adequately Insulated")   
        self.adqinsul_list.insert(3,"3. Poorly Insulated")
        self.adqinsul_list.insert(4,"4. No Insulation")   
        
        self.submit_button = Button(self, text="Submit", command=self.process)
        self.submit_button.grid(row=28, column=0, sticky=W)
        
        self.text = Text(self, width=35, height=5, wrap = WORD)
        self.text.grid(row=30, column=0, columnspan=2, sticky=NW)
        
    def process(self):
        walltype = sum(self.walltype_list.curselection(), 1)
        rooftype = sum(self.rooftype_list.curselection(), 1)
        yearmade = self.yearmade.get()
        region = self.region.get()
        bedrooms = self.bedrooms.get()
        adqinsul = sum(self.adqinsul_list.curselection(), 1)
        ### Prediction -- KWH_bin
        X3_new = np.array([[walltype, rooftype, yearmade, region, bedrooms, adqinsul]])
        prediction3 = knn3.predict(X3_new)
        prediction_text = "Prediction: %d KWH" % prediction3
        #print('Prediction:', prediction3,'KWH Range Mean')
        #prediction_text = "Walltype: %s" % walltype
        self.text.delete(0.0, END)
        self.text.insert(0.0, prediction_text)

########################################
# ENVIRONMENT PREP
import os

### Provide the path here
os.chdir('C:\\Users\\akash\\Desktop\\GWU\\6103_DataMining_FBradley\\Project_1') 

### Basic Packages
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.neighbors import KNeighborsClassifier

## DataLoad and Global Filtering

### Data Load
complete_energy_df  = pd.read_csv('recs2009_public2.csv')
keys_df = pd.read_csv('public_layout.csv')

### Create More Managable DataFrame
energy_df = complete_energy_df[['KWH','WALLTYPE','ROOFTYPE','YEARMADE','REGIONC','BEDROOMS','ADQINSUL','TYPEGLASS','WINDOWS','DOOR1SUM','FUELHEAT','TOTSQFT','AUDIT','NUMPC','NHSLDMEM','NUMFRIG','TVCOLOR','TOTROOMS']]

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

########################################

# DATAFRAME CREATION

## X-Variables
x_energy_df = energy_df.drop('KWH',axis=1,inplace=False)
x_feature_cols = list(x_energy_df)

## Y-Variable

### Y-Variable Categorization 
#### Create values to group the KWH energy usage into bins

conditions = [
        (energy_df['KWH'] >= 0) & (energy_df['KWH'] < 6000),
     (energy_df['KWH'] >= 6000) & (energy_df['KWH'] < 9500),
     (energy_df['KWH'] >= 9500) & (energy_df['KWH'] < 15000),
     (energy_df['KWH'] >= 15000)]
choices=[0,1,2,3]

energy_df['KWH_bin'] = np.select(conditions, choices, default=3)
y2_energy_df = energy_df[['KWH_bin']]

########################################

# MODEL-1 (DECISION TREE)

## Partition Data (DecisionTree)
X_train, X_test, Y_train, Y_test =tts(x_energy_df, y2_energy_df, test_size = 0.3, random_state=6103)

## Depth Determination (DecisionTree)
### Range of values to try, and where to store MSE output
max_depth_range = range(1, 12)
all_MSE_scores = []

### Calculate MSE for each value of max_depth
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=6103)
    MSE_scores = cross_val_score(treereg, X_train, Y_train, cv=14, scoring='neg_mean_squared_error')
    all_MSE_scores.append(np.mean(np.sqrt(-MSE_scores)))
    
### Plot max_depth (x-axis) versus MSE (y-axis)
plt.plot(max_depth_range, all_MSE_scores)
plt.title('Max Depth Range Plot (Decision Tree)')
plt.xlabel('max_depth')
plt.ylabel('MSE (lower is better)')
plt.savefig('DecisionTree_MaxDepthPlot.png')
plt.show()

## Feature Importance
### Based on max_depth plot, depth = 7 is most ideal
treereg = DecisionTreeRegressor(max_depth=7, random_state=6103)
treereg.fit(X_train, Y_train)


### "Gini importance" of each feature: 
print(pd.DataFrame({'feature':x_feature_cols, 'importance':sorted(treereg.feature_importances_ *1000, reverse = True)}))

del(all_MSE_scores,MSE_scores,depth)
########################################

# NEW DATAFRAME CREATION 
## Based on MODEL-1(Decision Tree)

x2_energy_df = x_energy_df[['WALLTYPE','ROOFTYPE','YEARMADE','REGIONC','BEDROOMS','ADQINSUL']].copy()

### Feature_Cols Based on Results from Decision Tree Model
feature_colsB = list(x2_energy_df)

## Partition Data (Post DecisionTree)
X2_train, X2_test, Y2_train, Y2_test =tts(x2_energy_df, y2_energy_df, test_size = 0.3, random_state=6103)

########################################

# EDA

## X-Variable Exploration

scatter_matrix(x2_energy_df, alpha=0.2, figsize=(12, 12), diagonal='kde')
plt.savefig('scatter_matrix1.png')
plt.show()

## Y-Variable Exploration

plt.hist(energy_df.KWH, bins=1000)
plt.grid = True
plt.savefig('KWH.png')
plt.show()

plt.hist(energy_df.KWH, bins=4)
plt.grid = True
plt.savefig('KWH2.png')
plt.show()

plt.hist(y2_energy_df.KWH_bin)
plt.grid = True
plt.savefig('KWHBin.png')
plt.show()

########################################

# MODEL-2 (KNN)

## KNN-Tuning -->
### Determines what number should K should be

### Store results
train_accuracy = []
test_accuracy  = []
### Set KNN setting from 1 to 15
knn2_range = range(1, 15)
for neighbors in knn2_range:
### Start Nearest Neighbors Classifier with K of 1
  knn2 = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski', p=1)
### Train the data using Nearest Neighbors
  knn2.fit(X2_train, Y2_train)
### Capture training accuracy
  train_accuracy.append(knn2.score(X2_train, Y2_train))
### Predict using the test dataset  
  Y_pred = knn2.predict(X2_test)
### Capture test accuracy
  test_accuracy.append(knn2.score(X2_test, Y2_test))
  
## Plot Results from KNN Tuning
plt.plot(knn2_range, train_accuracy, label='training accuracy')
plt.plot(knn2_range, test_accuracy,  label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Neighbors')
plt.legend()
plt.savefig('KNNTuning.png')
plt.show()

## KNN Using Output from KNN-Tuning, K = 9 is most ideal
knn2 = KNeighborsClassifier(n_neighbors=9, metric='minkowski', p=1)

### Re-Train the data using Nearest Neighbors
knn2.fit(X2_train, Y2_train)

### Model-2 Accuracy
Y2_pred= knn2.predict(X2_test)
print('\nPrediction from X_test:')
print(Y2_pred)

score2 = knn2.score(X2_test, Y2_test)
print('Score:', score2*100,'%')

### Prediction -- KWH_bin
X2_new = np.array([[3, 5,1995,3,3,3]])
prediction2 = knn2.predict(X2_new)
print('Prediction:', prediction2,'KWH bin')

########################################

# MODEL-3 (KNN)
## Repeating KNN replacing KWH_bin with KWH for a precise prediction output

## Partition Data (Post DecisionTree)
### Recreate Y-Variable with KWH (Instead of KWH_bin)
y3_energy_df = energy_df[['KWH']]
X3_train, X3_test, Y3_train, Y3_test =tts(x2_energy_df, y3_energy_df, test_size = 0.3, random_state=6103)

## KNN Using Output from KNN-Tuning, K = 9 is most ideal
knn3 = KNeighborsClassifier(n_neighbors=9, metric='minkowski', p=1)
### train the data using Nearest Neighbors
knn3.fit(X3_train, Y3_train)
Y3_pred= knn3.predict(X3_test)
print('\nPrediction from X_test:')
print(Y3_pred)
score3 = knn3.score(X3_test, Y3_test)
print('Score:', score3*100,'%')
X3_new = np.array([[3, 5,1995,3,3,3]])
prediction3 = knn3.predict(X3_new)
print('Prediction:', prediction3,'KWH')

########################################
del(complete_energy_df, keys_df)
del(X_train, X_test, Y_train, Y_test)
########################################

root = Tk()

root.title("simple gui")
root.geometry("400x610")
app = Application(root)
root.mainloop()

########################################
