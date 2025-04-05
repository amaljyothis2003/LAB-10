import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression

# Initialize Spark
def create_spark_session():
    return SparkSession.builder.master("local[*]").appName("WomensClothingReview").getOrCreate()

# Generate synthetic data
def generate_synthetic_data(num_rows):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, num_rows),
        'Rating': np.random.randint(1, 6, num_rows),
        'Positive Feedback Count': np.random.randint(0, 100, num_rows),
        'Recommended IND': np.random.randint(0, 2, num_rows)
    }
    return pd.DataFrame(data)

# Plot regression
def plot_regression(df_pandas):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df_pandas['Age'], y=df_pandas['Rating'], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.xlabel("Age")
    plt.ylabel("Rating")
    plt.title("Regression Line")
    st.pyplot(plt)

# Plot clusters
def plot_clusters(df_pandas, predictions):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_pandas['Age'], y=df_pandas['Positive Feedback Count'], hue=predictions.squeeze(), palette='viridis')
    plt.xlabel("Age")
    plt.ylabel("Positive Feedback Count")
    plt.title("K-Means Clusters")
    st.pyplot(plt)

# Title
st.title("Lab 10: Women's Clothing Reviews")

# Sidebar with slider
num_rows = st.sidebar.select_slider("Number of Records", options=[0, 10, 100, 500, 1000], value=500)
generate_btn = st.sidebar.button("Generate Dataset")

if generate_btn:
    df = generate_synthetic_data(num_rows)
    st.session_state.df = df
    st.session_state.spark = create_spark_session()
    st.session_state.spark_df = st.session_state.spark.createDataFrame(df)

if 'df' in st.session_state:
    df = st.session_state.df
    spark_df = st.session_state.spark_df

    st.subheader(f"Generated Dataset with {len(df)} Records")
    st.dataframe(df)

    # Sidebar options
    st.sidebar.markdown("---")
    clean_btn = st.sidebar.button("Clean Data")
    eda_btn = st.sidebar.button("Perform EDA")
    regression_btn = st.sidebar.button("Run Regression")
    cluster_btn = st.sidebar.button("Run Clustering")
    classify_btn = st.sidebar.button("Run Classification")

    # Cleaning
    if clean_btn:
        df_cleaned = df.dropna()
        st.session_state.df = df_cleaned
        st.session_state.spark_df = st.session_state.spark.createDataFrame(df_cleaned)
        st.success("Data cleaned successfully.")
        st.subheader("Cleaned Data")
        st.dataframe(df_cleaned)

    # EDA
    if eda_btn:
        pdf = st.session_state.spark_df.toPandas()
        fig, ax = plt.subplots()
        sns.histplot(pdf['Rating'], bins=5, ax=ax)
        plt.title("Distribution of Ratings")
        st.pyplot(fig)
        st.write("### Descriptive Statistics:")
        st.dataframe(st.session_state.spark_df.describe().toPandas())

    # Regression
    if regression_btn:
        assembler = VectorAssembler(inputCols=['Age', 'Rating'], outputCol='features')
        df_model = assembler.transform(st.session_state.spark_df).select('features', col('Rating').alias('label'))
        lr = LinearRegression()
        model = lr.fit(df_model)
        st.write("### Linear Regression")
        st.write("Coefficients:", model.coefficients)
        st.write("Intercept:", model.intercept)
        plot_regression(st.session_state.df)

    # Clustering
    if cluster_btn:
        assembler = VectorAssembler(inputCols=['Age', 'Positive Feedback Count'], outputCol='features')
        df_model = assembler.transform(st.session_state.spark_df).select('features')
        kmeans = KMeans(k=3)
        model = kmeans.fit(df_model)
        predictions = model.transform(df_model).select("prediction").toPandas()
        st.write("### K-Means Clustering")
        st.write("Cluster Centers:", model.clusterCenters())
        plot_clusters(st.session_state.df, predictions)

    # Classification
    if classify_btn:
        assembler = VectorAssembler(inputCols=['Age', 'Rating'], outputCol='features')
        df_model = assembler.transform(st.session_state.spark_df).select('features', col('Recommended IND').alias('label'))
        log_reg = LogisticRegression()
        model = log_reg.fit(df_model)
        st.write("### Logistic Regression (Classification)")
        st.write("Coefficients:", model.coefficients)
        st.write("Intercept:", model.intercept)

        # Predict for custom test inputs
        test_data = st.session_state.spark.createDataFrame([(30, 4), (50, 1)], ["Age", "Rating"])
        test_data = assembler.transform(test_data).select("features")
        predictions = model.transform(test_data).select("prediction").toPandas()
        st.write("Predictions:")
        st.write("Test 1 - Age: 30, Rating: 4 → Recommended:", int(predictions['prediction'][0]))
        st.write("Test 2 - Age: 50, Rating: 1 → Recommended:", int(predictions['prediction'][1]))
