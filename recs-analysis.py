
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier

# columns we care about
# WALLTYPE - Major outside wall material
# ROOFTYPE - Major roofing material
# YEARMADE - Year housing unit was built
# AIA_Zone - AIA Climate Zone, based on average temperatures from 1981 - 2010
# FUELHEAT - Main space heating fuel
# DNTAC    - No air conditioning equipment, or unused air conditioning equipment
# BEDROOMS - number of bedrooms in house
# WINDOWS  - number of windows in cooled/heated area
# KWH      - total KWH consumed for the year
# AUDIT    - whether an energy audit has been performed
# TOTSQFT  - total square footage of living space
# KWH_range - classification range of KWH consumed (0 thru 10)
#working_energy_df = energy_df[['WALLTYPE','ROOFTYPE','YEARMADE','AIA_Zone','FUELHEAT','DNTAC','BEDROOMS','WINDOWS','KWH','AUDIT','TOTSQFT','KWH_range']]
feature_cols = ['WALLTYPE','ROOFTYPE','YEARMADE','AIA_Zone','FUELHEAT','BEDROOMS','WINDOWS','TOTSQFT','KWH','ESWWAC','TYPEGLASS','CENACHP','NUMPC','TVCOLOR']
feature_cols_minus_target = ['WALLTYPE','ROOFTYPE','YEARMADE','AIA_Zone','FUELHEAT','BEDROOMS','WINDOWS','TOTSQFT','ESWWAC','TYPEGLASS','CENACHP','NUMPC','TVCOLOR']



# read in the 2009 survey data from CSV
complete_energy_df = pd.read_csv('data/recs2009_public.csv')
keys_df = pd.read_csv('data/public_layout.csv')

# copy dataframe columns to new dataframe
energy_df = complete_energy_df[feature_cols].copy()

# remove all records where negative values exist
energy_df = energy_df[(energy_df >= 0).all(1)]

# create KWH_range values to group the energy usage into ranges
conditions = [
    (energy_df['KWH'] >= 0) & (energy_df['KWH'] < 1000),
    (energy_df['KWH'] >= 2000) & (energy_df['KWH'] < 4000),
    (energy_df['KWH'] >= 4000) & (energy_df['KWH'] < 6000),
    (energy_df['KWH'] >= 6000) & (energy_df['KWH'] < 8000),
    (energy_df['KWH'] >= 8000) & (energy_df['KWH'] < 10000),
    (energy_df['KWH'] >= 10000) & (energy_df['KWH'] < 12000),
    (energy_df['KWH'] >= 12000) & (energy_df['KWH'] < 14000),
    (energy_df['KWH'] >= 14000) & (energy_df['KWH'] < 16000),
    (energy_df['KWH'] >= 16000) & (energy_df['KWH'] < 18000),
    (energy_df['KWH'] >= 18000) & (energy_df['KWH'] < 20000),
    (energy_df['KWH'] >= 20000)]
choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
energy_df['KWH_range'] = np.select(conditions, choices, default=10)





data_energy_df = energy_df[feature_cols_minus_target]
target_energy_df = energy_df[['KWH_range']]

# split data into training set & test set
X_train, X_test, Y_train, Y_test =tts(data_energy_df, target_energy_df, test_size = 0.3, random_state=6103)


# capture results
train_accuracy = []
test_accuracy  = []

# knn tuning --> What K should we pick
# set kNN setting from 1 to 9
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

# plot results
plt.plot(kNN_range, train_accuracy, label='training accuracy')
plt.plot(kNN_range, test_accuracy,  label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Neighbors')
plt.legend()

# start Nearest Neighbors Classifier with K of 1
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)

# train the data using Nearest Neighbors
knn.fit(X_train, Y_train)
















#pd.plotting.scatter_matrix(working_energy_df,
#                           figsize=(12, 12),
#                           marker='o',
#                           hist_kwds={'bins': 30},
#                           s=30,
#                           alpha=.8,
#                           cmap='winter')






from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score



############################################################
## calculate MSE score for each value of max_depth
## list of values to try
#max_depth_range = range(1, 8)
## list to store the average MSE for each value of max_depth
#all_MSE_scores = []

#for depth in max_depth_range:
#    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
#    MSE_scores = cross_val_score(treereg, X, y, cv=14, scoring='neg_mean_squared_error')
#    all_MSE_scores.append(np.mean(np.sqrt(-MSE_scores)))

## plot max_depth (x-axis) versus MSE (y-axis)
#plt.plot(max_depth_range, all_MSE_scores)
#plt.xlabel('max_depth')
#plt.ylabel('MSE (lower is better)')


# force the plot window to show
plt.show(block=True)


#treereg = DecisionTreeRegressor(max_depth=6, random_state=1)
#treereg.fit(X, y)

## "Gini importance" of each feature:
##    the (normalized) total reduction of error brought by that feature
#print(pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_}))


#from sklearn.tree import export_graphviz
#export_graphviz(treereg, out_file='tree_vehicles.dot', feature_names=feature_cols)
