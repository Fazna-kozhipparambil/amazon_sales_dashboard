import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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

# Handle missing values by filling them with a placeholder or dropping them
df.fillna(0, inplace=True)  # Replace null values with 0, or you can use another value

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

# Sales Trend Over Time
# We assume that the dataset does not have date-related columns, so we'll create an arbitrary time series for visualization purposes.
st.subheader('Sales Trend Over Time')
fig, ax = plt.subplots()
filtered_df['index'] = range(len(filtered_df))  # Create a pseudo time index
sns.lineplot(data=filtered_df, x='index', y='discounted_price', ax=ax)
ax.set_title('Sales Trend Over Time (Pseudo Time)')
st.pyplot(fig)
