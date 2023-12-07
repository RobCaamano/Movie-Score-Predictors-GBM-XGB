from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Preprocess Data
data = pd.read_csv('./archive/preprocessed_movies.csv')

# Splitting into train and test sets
X = data.drop('norm_score', axis=1)
y = data['norm_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# GBM model
model = GradientBoostingRegressor(
    n_estimators=100,
    validation_fraction=0.1, # Validation
    n_iter_no_change=25, # Early stopping
    tol=0.001, # Early stopping tolerance
    random_state=10, 
    verbose=1
)

# Test out different hyperparameters
param_grid = {
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [int(0.01 * len(X_train)), int(0.015 * len(X_train)), int(0.02 * len(X_train)), int(0.025 * len(X_train))]
}

# Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters found: ", grid_search.best_params_)

# Best parameters found:  {'max_depth': 15, 'min_samples_split': 65}