from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

digits = load_boston()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=1, population_size=10, verbosity=2)


tpot = TPOTRegressor(generations=1, population_size=10, verbosity=3,
                     # cv=GroupKFold(n_splits=5),
                     early_stop=5,
                     random_state=1,
                     )


tpot.fit(X, y)