import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier
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

        self.walltype_list = Listbox(self, selectmode=SINGLE, height=9)
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
        
        self.rooftype_list = Listbox(self, selectmode=SINGLE, height=8)
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
        self.adqinsul_text.grid(row=22, column=0, columnspan=1, sticky = NW)
        
        self.adqinsul_list = Listbox(self, selectmode=SINGLE, height=4)
        self.adqinsul_list.grid(row=22, column=1, rowspan=4, sticky=NW)
        self.adqinsul_list.insert(1,"Well Insulated")   
        self.adqinsul_list.insert(2,"Adequately Insulated")   
        self.adqinsul_list.insert(3,"Poorly Insulated")
        self.adqinsul_list.insert(4,"No Insulation")   
        
        self.submit_button = Button(self, text="Submit", command=self.process)
        self.submit_button.grid(row=27, column=0, sticky=W)
        
        self.text = Text(self, width=35, height=5, wrap = WORD)
        self.text.grid(row=29, column=0, columnspan=2, sticky=NW)
        
    def process(self):
        walltype = sum(self.walltype_list.curselection(), 1)
        rooftype = sum(self.rooftype_list.curselection(), 1)
        yearmade = self.yearmade.get()
        aia_zone = self.aia_zone.get()
        bedrooms = self.bedrooms.get()
        adqinsul = sum(self.adqinsul_list.curselection(), 1)
        ### Prediction -- KWH_Range
        #X_new = np.array([[walltype, rooftype, yearmade, aia_zone, bedrooms, adqinsul]])
        prediction = knn.predict(X_new)
        prediction_text = "Prediction: %d KWH Range Mean" % prediction
        #print('Prediction:', prediction,'KWH Range Mean')
        #prediction_text = "Walltype: %s" % walltype
        self.text.delete(0.0, END)
        self.text.insert(0.0, prediction_text)
        

########################################
# ENVIRONMENT PREP
import os
### Provide the path here
os.chdir('/Users/jimgrund/PycharmProjects/dats6103-recs/data/') 

### Basic Packages
import pandas as pd
import numpy as np

## DataLoad and Global Filtering

### read in the 2009 survey data from CSV 
complete_energy_df  = pd.read_csv('recs2009_public.csv')
keys_df = pd.read_csv('public_layout.csv')

### Create More Managable DataFrame
energy_df = complete_energy_df[['KWH','WALLTYPE','ROOFTYPE','YEARMADE','AIA_Zone','BEDROOMS','ADQINSUL','TYPEGLASS','WINDOWS','DOOR1SUM','FUELHEAT','TOTSQFT','AUDIT','NUMPC','NHSLDMEM','NUMFRIG','TVCOLOR','TOTROOMS']]

del(complete_energy_df)
del(keys_df)
#########################################################
# Create DataFrame for Model Data

data_energy_dfA = energy_df.drop('KWH',axis=1,inplace=False)

# Check for nulls (value = -2)
feature_colsA = list(data_energy_dfA)
nulls_df = pd.DataFrame()
i=0
while i < len(feature_colsA):
    print(feature_colsA[i])
    print((data_energy_dfA[feature_colsA[i]] ==-2).sum())
    nulls_df[feature_colsA[i]] = (data_energy_dfA[feature_colsA[i]] ==-2).sum()
    i+=1
del(nulls_df)
del(i)

# remove all records where nulls exist (value = -2)
#data_energy_df = data_energy_df[(data_energy_df >= 0).all(1)]

#########################################################
# Create KWH_range values to group the energy usage into ranges

data = energy_df['KWH']
kwh_min = min(energy_df.KWH)
kwh_max = max(energy_df.KWH)
kwh_bins = [np.linspace(kwh_min, kwh_max, 2000)]
kwh_bins = np.ravel(kwh_bins)
digitized = np.digitize(data, kwh_bins)
kwh_means = [data[digitized == i].mean() for i in range(1,len(kwh_bins))]
#^^ delete these two hashs 
labels = ['kwh_range_mean']
#kwh_bins = pd.DataFrame(kwh_bins, columns=labels)
kwh_bins = pd.DataFrame(kwh_means, columns = labels)

kwh_bins = kwh_bins[np.isfinite(kwh_bins['kwh_range_mean'])]

del(data)
del(kwh_min)
del(kwh_max)
del(labels)

target_energy_df = energy_df[['KWH']]
#replace with ~~>
#target_energy_df = energy_df[['KWH_range_mean']]

from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt

#########################################################
# DATA FILTERING 
### Create a Dataframe Based on Results from Decision Tree Model
data_energy_dfB = data_energy_dfA[['WALLTYPE','ROOFTYPE','YEARMADE','AIA_Zone','BEDROOMS','ADQINSUL']]
del(data_energy_dfA)

## Feature_Cols Based on Results from Decision Tree Model
feature_cols = list(data_energy_dfB)


## split data into training set & test set
X_train, X_test, Y_train, Y_test =tts(data_energy_dfB, target_energy_df, test_size = 0.3, random_state=6103)


from sklearn.neighbors import KNeighborsClassifier

## knn tuning --> What K should we pick
### capture results
train_accuracy = []
test_accuracy  = []
### set kNN setting from 1 to 11
kNN_range = range(1, 11)
for neighbors in kNN_range:
  # start Nearest Neighbors Classifier with K of 1
  knn = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski', p=2)
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

## start Nearest Neighbors Classifier with K of 7
knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)

### train the data using Nearest Neighbors
knn.fit(X_train, Y_train)

### Model accuracy
Y_pred= knn.predict(X_test)
print('\nPrediction from X_test:')
print(Y_pred)

score = knn.score(X_test, Y_test)
print('Score:', score*100,'%')

### Prediction -- KWH_Range
X_new = np.array([[2, 2, 2000,2,2,2]])
prediction = knn.predict(X_new)
print('Prediction:', prediction,'KWH Range Mean')



        
root = Tk()

root.title("simple gui")
root.geometry("400x610")
app = Application(root)
root.mainloop()

