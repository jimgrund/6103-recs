import pandas as pd
import matplotlib.pyplot as plt


#print(keys_df['Variable Name'])
#x = keys_df.T
#print(x)

#str_energy_df = energy_df.astype('str')
#frames = [str_energy_df, x]
#result=pd.concat(frames)


#results = pd.merge(str_energy_df, keys_df, on='DOEID')

#print(energy_df.dtypes)
#energy_df.apply(pd.to_object)
#print(energy_df.astype('str').dtypes)



# read in the 2009 survey data from CSV
energy_df = pd.read_csv('data/recs2009_public.csv')
keys_df = pd.read_csv('data/public_layout.csv')

# create a dataframe that contains only the columns we care about
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
working_energy_df = energy_df[['WALLTYPE','ROOFTYPE','YEARMADE','AIA_Zone','FUELHEAT','DNTAC','BEDROOMS','WINDOWS','KWH','AUDIT']]



#pd.plotting.scatter_matrix(working_energy_df,
#                           figsize=(12, 12),
#                           marker='o',
#                           hist_kwds={'bins': 30},
#                           s=30,
#                           alpha=.8,
#                           cmap='winter')





import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score

feature_cols = ['WALLTYPE','ROOFTYPE','YEARMADE','AIA_Zone','FUELHEAT','DNTAC','BEDROOMS','WINDOWS','AUDIT']
X = working_energy_df[feature_cols]
y = working_energy_df.KWH


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
