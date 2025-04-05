import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans

# Streamlit UI
st.title("Lab 10: Women's Clothing Reviews")

# Function to generate synthetic data
def generate_synthetic_data(num_rows=500):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, num_rows),
        'Rating': np.random.randint(1, 6, num_rows),
        'Positive Feedback Count': np.random.randint(0, 100, num_rows),
        'Recommended IND': np.random.randint(0, 2, num_rows)
    }
    return pd.DataFrame(data)

# Plotting functions
def plot_regression(df, model):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df['Age'], y=df['Rating'], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.xlabel("Age")
    plt.ylabel("Rating")
    plt.title("Regression Line")
    st.pyplot(plt)

def plot_clusters(df, predictions):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['Age'], y=df['Positive Feedback Count'], hue=predictions, palette='viridis')
    plt.xlabel("Age")
    plt.ylabel("Positive Feedback Count")
    plt.title("K-Means Clusters")
    st.pyplot(plt)

# Sidebar: Generate data
st.sidebar.subheader("Synthetic Data Generator")
num_rows = st.sidebar.slider("Number of Records", 0,10,100, 1000, 500)
generate_btn = st.sidebar.button("Generate Dataset")

# Session state to store data
if "df" not in st.session_state:
    st.session_state.df = None

if generate_btn:
    df = generate_synthetic_data(num_rows)
    st.session_state.df = df
    st.success("Synthetic dataset generated!")

# Show options only if data is available
if st.session_state.df is not None:
    df = st.session_state.df
    st.write("### Data Sample:")
    st.write(df)

    st.sidebar.subheader("Operations")

    if st.sidebar.button("Clean Data"):
        df = df.dropna()
        st.session_state.df = df
        st.write("### Cleaned Data Sample:")
        st.write(df)

    if st.sidebar.button("Perform EDA"):
        st.write("### Exploratory Data Analysis")
        fig, ax = plt.subplots()
        sns.histplot(df['Rating'], bins=5, ax=ax)
        st.pyplot(fig)

        st.write("### Summary Statistics")
        st.write(df.describe())

    if st.sidebar.button("Run Regression"):
        model = LinearRegression()
        X = df[['Age']]
        y = df['Rating']
        model.fit(X, y)
        st.write("Regression Coefficients:", model.coef_)
        st.write("Intercept:", model.intercept_)
        plot_regression(df, model)

    if st.sidebar.button("Run Clustering"):
        model = KMeans(n_clusters=3, random_state=42)
        X = df[['Age', 'Positive Feedback Count']]
        predictions = model.fit_predict(X)
        st.write("Cluster Centers:", model.cluster_centers_)
        plot_clusters(df, predictions)

    if st.sidebar.button("Run Classification"):
        model = LogisticRegression()
        X = df[['Age', 'Rating']]
        y = df['Recommended IND']
        model.fit(X, y)
        st.write("Classification Coefficients:", model.coef_)
        st.write("Intercept:", model.intercept_)

        # Predict on test data
        test_data = pd.DataFrame({'Age': [30, 50], 'Rating': [4, 1]})
        predictions = model.predict(test_data)
        st.write("Test 1 - Age: 30, Rating: 4 → Prediction:", predictions[0])
        st.write("Test 2 - Age: 50, Rating: 1 → Prediction:", predictions[1])
