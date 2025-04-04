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

def create_spark_session():
    return SparkSession.builder.appName("WomensClothingReview").getOrCreate()

def generate_synthetic_data(num_rows=500):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, num_rows),
        'Rating': np.random.randint(1, 6, num_rows),
        'Positive Feedback Count': np.random.randint(0, 100, num_rows),
        'Recommended IND': np.random.randint(0, 2, num_rows)
    }
    return pd.DataFrame(data)

def plot_regression(df_pandas, model):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df_pandas['Age'], y=df_pandas['Rating'], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.xlabel("Age")
    plt.ylabel("Rating")
    plt.title("Regression Line")
    st.pyplot(plt)

def plot_clusters(df_pandas, predictions):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_pandas['Age'], y=df_pandas['Positive Feedback Count'], hue=predictions.squeeze(), palette='viridis')
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
    spark = create_spark_session()
    df = spark.createDataFrame(df_pandas)
    
    st.write("### Data Sample:")
    st.write(df.limit(50).toPandas())

    # Clean Data
    if st.button("Clean Data"):
        df_pandas = df_pandas.dropna()
        df = spark.createDataFrame(df_pandas)
        st.write("### Cleaned Data Sample:")
        st.write(df.limit(50).toPandas())

    # EDA
    if st.button("Perform EDA"):
        pdf = df.toPandas()
        fig, ax = plt.subplots()
        sns.histplot(pdf['Rating'], bins=5, ax=ax)
        st.pyplot(fig)
        
        st.write("### More EDA:")
        st.write(df.describe().toPandas())

    # Regression
    if st.button("Run Regression"):
        assembler = VectorAssembler(inputCols=['Age', 'Rating'], outputCol='features')
        df_model = assembler.transform(df).select('features', col('Rating').alias('label'))
        lr = LinearRegression()
        model = lr.fit(df_model)
        st.write("Regression Model Coefficients:", model.coefficients)
        st.write("Intercept:", model.intercept)
        plot_regression(df_pandas, model)

    # Clustering
    if st.button("Run Clustering"):
        assembler = VectorAssembler(inputCols=['Age', 'Positive Feedback Count'], outputCol='features')
        df_model = assembler.transform(df).select('features')
        kmeans = KMeans(k=3)
        model = kmeans.fit(df_model)
        predictions = model.transform(df_model).select("prediction").toPandas()
        st.write("Cluster Centers:", model.clusterCenters())
        plot_clusters(df_pandas, predictions)

    # Classification
    if st.button("Run Classification"):
        assembler = VectorAssembler(inputCols=['Age', 'Rating'], outputCol='features')
        df_model = assembler.transform(df).select('features', col('Recommended IND').alias('label'))
        log_reg = LogisticRegression()
        model = log_reg.fit(df_model)
        st.write("Classification Model Coefficients:", model.coefficients)
        st.write("Intercept:", model.intercept)
        
        test_data = spark.createDataFrame([(30, 4), (50, 1)], ["Age", "Rating"])
        test_data = assembler.transform(test_data).select("features")
        predictions = model.transform(test_data).select("prediction").toPandas()
        st.write("Test 1 - Age: 30, Rating: 4")
        st.write("Test 2 - Age: 50, Rating: 1")
        st.write("Predictions - Recommended IND:", predictions)
