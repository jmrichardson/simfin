

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
sel.fit(X_train)
index = sel.get_support()

new = X_train.iloc[:,index].columns
columns = X_train.iloc[:,~index].columns

print(X_train.shape)
print(new.shape)