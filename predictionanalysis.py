# Import and setup libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Necessary backend setting
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # switch backend if only plt imported

import seaborn as sns
sns.set(style='whitegrid', palette='muted', color_codes=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Ensure inline plotting (Kaggle specific)
%matplotlib inline

# Set a random seed for reproducibility
RANDOM_STATE = 42

print('Libraries imported and backend configured.')

# Load the dataset
file_path = '/kaggle/input/temperature/temperature.csv'
try:
    df = pd.read_csv(file_path, encoding='ascii', delimiter=',')
    print('Data loaded successfully, sample records below:')
    display(df.head())
except Exception as error:
    print('Error loading the dataset. Check file path or encoding settings.')
    print(error)

    # Basic data exploration
print('Dataset shape:', df.shape)
print('\nDataset info:')
df.info()

# Check for missing values
print('\nMissing values per column:')
print(df.isnull().sum())

# Handle missing values if any (using median imputation for numeric columns as an example)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f'Filled missing values in {col} with median value {median_val}')

# A quick check of data types
print('\nData types after cleaning:')
print(df.dtypes)

# Descriptive statistics
print('Descriptive statistics for numerical features:')
display(df.describe())

# Correlation analysis on numeric data
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.show()
else:
    print('Not enough numeric columns for a correlation heatmap.')

# Pairplot for a subset of numeric columns to inspect bivariate relationships
sample_cols = [col for col in ['Year', 'Avg_Temperature_degC', 'CO2_Emissions_tons_per_capita',
                               'Sea_Level_Rise_mm', 'Rainfall_mm'] if col in numeric_df.columns]
if len(sample_cols) >= 2:
    sns.pairplot(df[sample_cols].dropna())
    plt.suptitle('Pairplot of Selected Numeric Features', y=1.02)
    plt.show()
else:
    print('Not enough variables for a meaningful pairplot.')

    # Histogram of Average Temperature
plt.figure(figsize=(8, 5))
sns.histplot(df['Avg_Temperature_degC'].dropna(), kde=True, color='skyblue')
plt.title('Distribution of Average Temperature (°C)')
plt.xlabel('Avg_Temperature_degC')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Box Plot for Avg Temperature by Country
plt.figure(figsize=(12, 6))
sns.boxplot(x='Country', y='Avg_Temperature_degC', data=df)
plt.title('Average Temperature by Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Count plot for Extreme Weather Events
plt.figure(figsize=(8, 5))
sns.countplot(x='Extreme_Weather_Events', data=df, palette='viridis')
plt.title('Count of Extreme Weather Events')
plt.xlabel('Number of Extreme Weather Events')
plt.tight_layout()
plt.show()

# Grouped Barplot: Renewable Energy vs. Forest Area by Country (aggregated)
agg_df = df.groupby('Country')[['Renewable_Energy_pct', 'Forest_Area_pct']].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='Country', y='Renewable_Energy_pct', data=agg_df, color='teal', label='Renewable Energy (%)')
sns.barplot(x='Country', y='Forest_Area_pct', data=agg_df, color='salmon', label='Forest Area (%)')
plt.title('Average Renewable Energy and Forest Area Percentage by Country')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# Prepare data for prediction
target = 'Avg_Temperature_degC'

# Choose features: excluding Country since it's categorical, although we could encode it
features = ['Year', 'CO2_Emissions_tons_per_capita', 'Sea_Level_Rise_mm',
            'Rainfall_mm', 'Population', 'Renewable_Energy_pct',
            'Extreme_Weather_Events', 'Forest_Area_pct']

# Drop rows with missing target or features
model_df = df[features + [target]].dropna()

# Define X and y
X = model_df[features]
y = model_df[target]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print('Data split into training and test sets.')

# Instantiate and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Linear Regression R² score: {r2:.3f}')
print(f'Linear Regression RMSE: {rmse:.3f}')

# Plot actual vs predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color='navy')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Average Temperature')
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.tight_layout()
plt.show()