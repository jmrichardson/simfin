

# Pandas is used for data manipulation
import pandas as pd
# Read in data and display first 5 rows

features = pd.read_csv('data/temps.csv')
features.head(5)
features = pd.get_dummies(features)


# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(features['actual'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


rf = RandomForestRegressor(n_estimators=10, n_jobs=-1)
scores = cross_val_score(rf, train_features, train_labels, cv=5, scoring='r2')






gkf = GroupKFold(n_splits=5)


scores = cross_val_score(rf, X, y, cv=gkf, groups=groups, scoring='r2')

