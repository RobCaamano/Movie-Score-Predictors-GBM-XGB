import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import xgboost as xgb
import optuna
import pickle

# Preprocess Data
data = pd.read_csv('./archive/preprocessed_movies.csv')

# Splitting into train and test sets
X = data.drop('norm_score', axis=1)
y = data['norm_score']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=10)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=10)

def objective(trial):
    # Tuning hyperparameters
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1]),
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 500, 1000, 1500]),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_loguniform('gamma', 1e-3, 2.0),
    }
    
    # Model training and evaluation
    model = xgb.XGBRegressor(**param)
    
    # Add a callback for pruning (automatically stops training if performance worsens)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        early_stopping_rounds=50, 
        callbacks=[pruning_callback],
        verbose=False
    )
    
    # Evaluation
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse

# Creating the Optuna study object
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best hyperparameters
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# Train the final model
best_params = study.best_trial.params
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Save the model to a file
with open('optuna_xgboost_model.pkl', 'wb') as model_file:
    pickle.dump(final_model, model_file)