import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv('data/housing.csv')

# Remove useless columns
df = df.drop(['ID', 'Date'], axis=1)

# Feature Engineering
df['Total_sqft'] = df['Sqft_living'] + df['Sqft_basement']
df['House_age'] = 2025 - df['Yr_built'] 

df = df.drop(['Yr_built'], axis=1)
df = df[df['Price'] < df['Price'].quantile(0.99)]

# Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print(f"Mean Absolute Error: {mae}")
