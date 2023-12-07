from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Preprocess Data
data = pd.read_csv('./archive/preprocessed_movies.csv')

# Splitting into train and test sets
X = data.drop('norm_score', axis=1)
y = data['norm_score']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=10)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=10)

# Warmup model
model = GradientBoostingRegressor(
    warm_start=True,
    n_estimators=50, 
    max_depth=15,
    min_samples_split=65,
    random_state=10, 
    verbose=1
)

# Training model (Early Stopping w/ Best Model)
best_mse = float("inf")
best_model = None
no_improvement_count=0

for i in range(1, 301):
    model.n_estimators += i
    model.fit(X_train, y_train)
    
    mse = mean_squared_error(y_val, model.predict(X_val))
    if mse < best_mse:
        best_mse = mse
        best_model = pickle.dumps(model)  # Save best model
        no_improvement_count = 0 
    else:
        no_improvement_count += 1

    if no_improvement_count >= 5:
        print(f"Stopping early after {i} iterations.")
        break

# Load best model
model = pickle.loads(best_model) if best_model is not None else model

# Evaluate the model's performance
print("Validation set R-squared:", r2_score(y_val, model.predict(X_val)))

train_errors = []
val_errors = []

# Iterate over each stage's predictions and record the MSE
for y_train_pred, y_val_pred in zip(model.staged_predict(X_train), model.staged_predict(X_val)):
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))

y_pred = model.predict(X_test)

# Scatter plot for Actual vs. Predicted values
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted norm_score (Gradient Boosting)')
plt.show()

# Training vs Testing Error
plt.figure(figsize=(10, 6))
plt.plot(train_errors, label='Training Error')
plt.plot(val_errors, label='Validation Error')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation Error by Number of Estimators')
plt.legend()
plt.show()

print("Best Validation MSE:", best_mse)
print("Train Error:", train_errors[-1])
print("Validation Error:", val_errors[-1])

# Save the model to a file
with open('gradient_boosting_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model saved successfully!")

#Validation set R-squared: 0.31381528617015975
#Best Validation MSE: 0.07725064103933495
#Train Error: 0.020565040625028726
#Validation Error: 0.07725064103933495