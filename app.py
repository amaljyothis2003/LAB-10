import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans

# Generate synthetic data
def generate_synthetic_data(num_rows=500):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, num_rows),
        'Rating': np.random.randint(1, 6, num_rows),
        'Positive Feedback Count': np.random.randint(0, 100, num_rows),
        'Recommended IND': np.random.randint(0, 2, num_rows)
    }
    return pd.DataFrame(data)

# Plot regression
def plot_regression(df_pandas, model):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df_pandas['Age'], y=df_pandas['Rating'], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.xlabel("Age")
    plt.ylabel("Rating")
    plt.title("Regression Line")
    st.pyplot(plt)

# Plot clusters
def plot_clusters(df_pandas, predictions):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_pandas['Age'], y=df_pandas['Positive Feedback Count'], hue=predictions, palette='viridis')
    plt.xlabel("Age")
    plt.ylabel("Positive Feedback Count")
    plt.title("K-Means Clusters")
    st.pyplot(plt)

# Streamlit UI
st.title("Lab 10: Women's Clothing Reviews")

# Generate data instead of uploading
st.subheader("Synthetic Data Generation")
num_rows = st.slider("Number of Records", 100, 1000, 500)

if st.button("Generate Dataset"):
    df_pandas = generate_synthetic_data(num_rows)
    st.session_state["df_pandas"] = df_pandas
    st.write("### Data Sample:")
    st.write(df_pandas.head(50))

# Use session state to persist data
if "df_pandas" in st.session_state:
    df_pandas = st.session_state["df_pandas"]

    # Clean Data
    if st.button("Clean Data"):
        df_pandas = df_pandas.dropna()
        st.session_state["df_pandas"] = df_pandas
        st.write("### Cleaned Data Sample:")
        st.write(df_pandas.head(50))

    # EDA
    if st.button("Perform EDA"):
        fig, ax = plt.subplots()
        sns.histplot(df_pandas['Rating'], bins=5, ax=ax)
        st.pyplot(fig)

        st.write("### More EDA:")
        st.write(df_pandas.describe())

    # Regression
    if st.button("Run Regression"):
        X = df_pandas[['Age']]
        y = df_pandas['Rating']
        model = LinearRegression()
        model.fit(X, y)

        st.write("Regression Coefficient:", model.coef_[0])
        st.write("Intercept:", model.intercept_)
        plot_regression(df_pandas, model)

    # Clustering
    if st.button("Run Clustering"):
        X = df_pandas[['Age', 'Positive Feedback Count']]
        kmeans = KMeans(n_clusters=3, random_state=0)
        predictions = kmeans.fit_predict(X)
        st.write("Cluster Centers:", kmeans.cluster_centers_)
        plot_clusters(df_pandas, predictions)

    # Classification
    if st.button("Run Classification"):
        X = df_pandas[['Age', 'Rating']]
        y = df_pandas['Recommended IND']
        model = LogisticRegression()
        model.fit(X, y)

        st.write("Classification Coefficients:", model.coef_)
        st.write("Intercept:", model.intercept_)

        test_data = pd.DataFrame([[30, 4], [50, 1]], columns=["Age", "Rating"])
        predictions = model.predict(test_data)
        st.write("Test 1 - Age: 30, Rating: 4")
        st.write("Test 2 - Age: 50, Rating: 1")
        st.write("Predictions - Recommended IND:", predictions)
