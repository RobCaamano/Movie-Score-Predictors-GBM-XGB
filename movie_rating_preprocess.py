import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

## Preprocess Data
data_raw = pd.read_csv('./archive/inference.csv')

# Visualize
print("First 5 Rows:")
data_raw.head() # View first 5 rows
print("\n\n")

print("Dimensions: ", data_raw.shape)
num_row = data_raw.shape[0]
num_col = data_raw.shape[1]
print(data_raw.dtypes)
print("\n\n")

data = data_raw.copy()

# Remove : country, name
data.drop(['country', 'name', 'year', 'company'], axis=1, inplace=True)

# Visualize NaN & Unique values
print("NaN Values:")
print(data.isna().sum())
print("Out of",num_row,"rows")
print("\n")

print("Unique Values:")
print(data.nunique())
print("Out of",num_row,"rows")
print("\n\n")

# Handle missing values by deleting rows
data.dropna(inplace=True)

# Create: score/votes (Predicted Value)
C = data['score'].mean()
m = data['votes'].quantile(0.9)

print("C:", C)
print("m:", m)
print("\n\n")

def weighted_rating(x, m=m, C=C):
    v = x['votes']
    R = x['score']
    return (v/(v+m) * R) + (m/(m+v) * C)

data['norm_score'] = data.apply(weighted_rating, axis=1)

# Keep: budget, director, genre, gross, rating, released, runtime, star, writer, year, norm_score
keep = ['budget', 'director', 'genre', 'gross', 
                   'rating', 'released', 'runtime', 'star', 'writer', 'norm_score']
data = data[keep]

# Released column feature engineering
data[['release_date', 'release_country']] = data['released'].str.extract(r'(.*) \((.*)\)')

data.dropna(subset=['release_date'], inplace=True)

data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

data.dropna(subset=['release_date'], inplace=True) # Drop rows with invalid dates

data['release_month'] = data['release_date'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'
    
data['season'] = data['release_month'].apply(get_season)

data.drop(['released', 'release_date', 'release_month', 'release_country'], axis=1, inplace=True)

## Plots
numeric_columns = ['budget', 'gross', 'runtime', 'norm_score']

# norm_score distribution
sns.histplot(data['norm_score'], kde=True)
plt.title('Distribution of norm_score')
plt.xlabel('norm_score')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix for numeric columns
correlation_matrix = data[numeric_columns].corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Correlations for categorical columns
categorical_columns = ['director', 'genre', 'star', 'writer', 'season']
anova_results = {}
for col in categorical_columns:
    grouped_data = [data['norm_score'][data[col] == category] for category in data[col].unique()]
    f_value, p_value = stats.f_oneway(*grouped_data)
    anova_results[col] = (f_value, p_value)

for col, (f_value, p_value) in anova_results.items():
    print(f"{col} - ANOVA F-value: {f_value}, P-value: {p_value}")

print("\n\n")

## Encoding / Scaling

# One hot encoding
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
encoded = one_hot_encoder.fit_transform(data[categorical_columns])
data = data.join(pd.DataFrame(encoded.toarray(), columns=one_hot_encoder.get_feature_names_out()))
data.fillna(0, inplace=True)

# Drop original categorical columns
data.drop(categorical_columns, axis=1, inplace=True)

# Label encode rating (ordinal)
label_encoder = LabelEncoder()
data['rating'] = label_encoder.fit_transform(data['rating'])

# Scaling
numeric_columns = ['budget', 'gross', 'runtime']
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Make sure all columns are numeric
for col in data.columns:
    if data[col].dtype == 'bool':
        data[col] = data[col].astype(int)

# Visualize
print("Data Types:")
print(data.dtypes)
print("\n")

print("NaN Values:")
print(data.isna().sum())
print("Out of",num_row,"rows")
print("\n\n")

data.to_csv('./archive/preprocessed_inference2.csv', index=False)

#director - ANOVA F-value: 1.7871584194117967, P-value: 1.540266080256144e-50
#genre - ANOVA F-value: 19.069844430726548, P-value: 1.6151122210114792e-47
#star - ANOVA F-value: 1.1593424013670217, P-value: 0.00012001780490810191
#writer - ANOVA F-value: 1.1096612444463727, P-value: 0.004045028064136678
#season - ANOVA F-value: 6.369862195874755, P-value: 0.0002634754935325735
#rating - ANOVA F-value: 4.417266387210736, P-value: 8.845628455830684e-06