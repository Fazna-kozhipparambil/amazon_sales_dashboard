import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

# Load the data
st.title("Amazon Sales Dashboard")
st.write("A comprehensive dashboard to explore and analyze Amazon sales data.")
file_path = 'amazon.csv'  # Update this if needed
df = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
st.subheader("Data Overview")
df['discounted_price'] = pd.to_numeric(df['discounted_price'].str.replace(',', '').str.replace('₹', ''), errors='coerce')
df['actual_price'] = pd.to_numeric(df['actual_price'].str.replace(',', '').str.replace('₹', ''), errors='coerce')
df['discount_percentage'] = pd.to_numeric(df['discount_percentage'].str.replace('%', ''), errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')

df.dropna(inplace=True)  # Drop rows with missing values

st.write("Cleaned Data")
st.write(df.head())

# Price Distribution
st.subheader('Price Distribution')
fig, ax = plt.subplots()
sns.histplot(df['discounted_price'], bins=30, ax=ax)
ax.set_title('Distribution of Discounted Prices')
st.pyplot(fig)

# Discount Percentage vs Rating
st.subheader('Discount vs Rating')
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='discount_percentage', y='rating', ax=ax)
ax.set_title('Discount Percentage vs Rating')
st.pyplot(fig)

# Average Rating by Category
st.subheader('Average Rating by Category')
avg_rating_by_category = df.groupby('category')['rating'].mean().sort_values(ascending=False)
st.bar_chart(avg_rating_by_category)

# Filter by Category
st.subheader('Filter by Category')
category = st.selectbox('Select Category', df['category'].unique())
filtered_df = df[df['category'] == category]

st.write(f"Showing data for category: {category}")
st.write(filtered_df[['product_name', 'discounted_price', 'actual_price', 'discount_percentage', 'rating']])

# Average Rating for Selected Category
st.subheader(f'Average Rating for {category}')
avg_rating = filtered_df['rating'].mean()
st.write(f'Average Rating: {avg_rating:.2f}')

# Sales Forecasting using Prophet
st.subheader('Sales Forecasting')

# Prepare the data for Prophet (Using discounted price as proxy for sales)
df_prophet = filtered_df[['product_name', 'discounted_price']].groupby('product_name').sum().reset_index()
df_prophet.rename(columns={'product_name': 'ds', 'discounted_price': 'y'}, inplace=True)

# Initialize the Prophet model
model = Prophet()
model.fit(df_prophet)

# Future dataframe for next 365 days
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot the forecast
fig2 = model.plot(forecast)
st.pyplot(fig2)

st.write("Forecast Data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Filter by Date Range
st.subheader('Filter by Date Range')
start_date = st.date_input('Start date', df['product_name'].min())
end_date = st.date_input('End date', df['product_name'].max())
filtered_df = df[(df['product_name'] >= pd.to_datetime(start_date)) & (df['product_name'] <= pd.to_datetime(end_date))]

st.write(f"Showing data from {start_date} to {end_date}")
st.write(filtered_df)
