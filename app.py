import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/housing.csv')
df = df.drop(['ID', 'Date'], axis=1)

# Feature engineering
df['Total_sqft'] = df['Sqft_living'] + df['Sqft_basement']
df['House_age'] = 2025 - df['Yr_built']
df = df.drop(['Yr_built'], axis=1)

# Remove outliers
df = df[df['Price'] < df['Price'].quantile(0.99)]

# Encoding
df = pd.get_dummies(df, drop_first=True)

# Train model
X = df.drop('Price', axis=1)
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# UI
st.title("🏠 House Price Prediction")

sqft_living = st.number_input("Living Area (sqft)", value=1000)
bedrooms = st.number_input("Bedrooms", value=2)
bathrooms = st.number_input("Bathrooms", value=2)

if st.button("Predict Price"):
    sample = X.iloc[0:1].copy()
    sample['Sqft_living'] = sqft_living
    sample['Bedrooms'] = bedrooms
    sample['Bathrooms'] = bathrooms
    sample['Total_sqft'] = sample['Sqft_living'] + sample['Sqft_basement']

    pred = model.predict(sample)[0]

    # USD → INR
    inr = pred * 83

    # Format
    if inr >= 10000000:
        result = f"₹{inr/10000000:.2f} Crore"
    else:
        result = f"₹{inr/100000:.2f} Lakh"

    st.success(f"Estimated Price: {result}")