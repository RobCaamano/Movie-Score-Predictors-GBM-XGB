from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

# Preprocess Data
data = pd.read_csv('./archive/preprocessed_movies.csv')

# Splitting into train and test sets
X = data.drop('norm_score', axis=1)
y = data['norm_score']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=10)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=10)

# Model
xgb_model = xgb.XGBRegressor(
    tree_method='gpu_hist',
    objective='reg:squarederror', 
    eval_metric='rmse',
    n_estimators=1000,
    reg_lambda=0.01026801452971141,
    reg_alpha=0.23821695120486366,
    colsample_bytree=1.0,
    subsample=0.9,
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=11,
    gamma=0.0033266288138543397,
    early_stopping_rounds=10,
    random_state=20
)

#Best trial: {'lambda': 0.01026801452971141, 'alpha': 0.23821695120486366, 'colsample_bytree': 1.0, 'subsample': 0.9, 'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 4, 'min_child_weight': 11, 'gamma': 0.0033266288138543397}

eval_set = [(X_train, y_train), (X_val, y_val)]

# Training model (Early Stopping)
xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=True 
)

# Best iteration
print("Best iteration:", xgb_model.best_iteration)
eval_result = xgb_model.evals_result()

# Predict off test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")

# Plotting the actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted norm_score (XGBoost)')
plt.show()

# Plotting the learning curves
plt.figure(figsize=(10, 6))
plt.plot(eval_result['validation_0']['rmse'], label='Train Error')
plt.plot(eval_result['validation_1']['rmse'], label='Val Error')
plt.xlabel('Number of Trees')
plt.ylabel('RMSE')
plt.title('XGBoost Training and Validation Error over Boosting Rounds')
plt.legend()
plt.show()

print("Train Error", eval_result['validation_0']['rmse'][-1])
print("Val Error", mean_squared_error(y_test, y_pred, squared=False))

# Save the model to a file
with open('xgboost_model.pkl', 'wb') as model_file:
    pickle.dump(xgb_model, model_file)

print("Model saved successfully!")

#Mean Squared Error (MSE): 0.049126765046695275
#Root Mean Squared Error (RMSE): 0.22164558431580647
#R^2 Score: 0.31015613350934423
#Train Error 0.22772960734535402
#Val Error 0.22164558431580647