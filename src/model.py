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

# -------- USER INPUT --------

print("\nEnter details of the house:")

sqft_living = float(input("Living area (sqft): "))
bedrooms = int(input("Number of bedrooms: "))
bathrooms = float(input("Number of bathrooms: "))

# Create input data (copy structure)
sample_house = X.iloc[0:1].copy()

# Update with user values
sample_house['Sqft_living'] = sqft_living
sample_house['Bedrooms'] = bedrooms
sample_house['Bathrooms'] = bathrooms

# Recalculate engineered features
sample_house['Total_sqft'] = sample_house['Sqft_living'] + sample_house['Sqft_basement']
sample_house['House_age'] = sample_house['House_age']  # keep same

def format_inr(price):
    if price >= 10000000:  # 1 Crore
        return f"₹{price/10000000:.2f} Crore"
    elif price >= 100000:  # 1 Lakh
        return f"₹{price/100000:.2f} Lakh"
    else:
        return f"₹{price:,.0f}"

predicted_price = model.predict(sample_house)

# Convert USD → INR (approx rate)
usd_price = predicted_price[0]
inr_price = usd_price * 83  # 1 USD ≈ ₹83

price = int(inr_price)
formatted_price = format_inr(price)

print(f"\n🏠 Estimated House Price: {formatted_price}")


