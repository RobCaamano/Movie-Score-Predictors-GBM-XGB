import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")

# Original Data
training_data = pd.read_csv('./archive/movies.csv')
data = pd.read_csv('./archive/inference.csv')

combined_data = pd.concat([training_data, data])

# Calculate norm_score for each row
C = combined_data['score'].mean()
m = combined_data['votes'].quantile(0.9)

def weighted_rating(x, m=m, C=C):
    v = x['votes']
    R = x['score']
    return (v/(v+m) * R) + (m/(m+v) * C)

data['norm_score'] = data.apply(weighted_rating, axis=1)
norm_scores_list = data['norm_score'].tolist()

# Preprocessed Inference Data
data_processed = pd.read_csv('./archive/preprocessed_inference.csv')

# Preprocessed Training Data merged with Inference Data
training_data_processed = pd.read_csv('./archive/preprocessed_movies.csv')
model_columns = training_data_processed.columns.tolist()

full_df = pd.DataFrame(data_processed, columns=model_columns).fillna(0)

full_df.drop(['norm_score'], axis=1, inplace=True)

## Load GBM model
print("GBM model")
with open('gradient_boosting_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Use 'loaded_model' for inference
predictions = loaded_model.predict(full_df)

# Round predictions
norm_scores_list = [round(num, 2) for num in norm_scores_list]
predictions = [round(num, 2) for num in predictions]
average_error = np.mean(np.abs(np.array(norm_scores_list) - np.array(predictions)))

print("Norm Scores: ", norm_scores_list)
print("Predictions: ", predictions)
print("Average Error: ", round(average_error, 2))

## Load XGB model
print("XGB Model")
with open('xgboost_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Use 'loaded_model' for inference
predictions = loaded_model.predict(full_df)

# Round predictions
norm_scores_list = [round(num, 2) for num in norm_scores_list]
predictions = [round(num, 2) for num in predictions]
average_error = np.mean(np.abs(np.array(norm_scores_list) - np.array(predictions)))

print("Norm Scores: ", norm_scores_list)
print("Predictions: ", predictions)
print("Average Error: ", round(average_error, 2))