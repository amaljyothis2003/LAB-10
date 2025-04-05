import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create Spark Session
def create_spark_session():
    return SparkSession.builder.appName("WomensClothingReview").getOrCreate()

# Function to generate synthetic dataset
def generate_synthetic_data(num_rows=10000):
    np.random.seed(42)
    data = {
        "Clothing ID": np.random.randint(1000, 1100, size=num_rows),
        "Age": np.random.randint(18, 70, size=num_rows),
        "Title": np.random.choice(["Great fit", "Love it", "Too small", "Not as expected", "Perfect"], size=num_rows),
        "Review Text": np.random.choice([
            "Loved it!", 
            "Too tight around the waist.", 
            "Color was off.", 
            "Just what I needed.", 
            "Material feels cheap."
        ], size=num_rows),
        "Rating": np.random.randint(1, 6, size=num_rows),
        "Recommended IND": np.random.choice([0, 1], size=num_rows),
        "Positive Feedback Count": np.random.poisson(5, size=num_rows),
        "Division Name": np.random.choice(["General", "Petites", "Intimates"], size=num_rows),
        "Department Name": np.random.choice(["Tops", "Dresses", "Bottoms", "Intimate"], size=num_rows),
        "Class Name": np.random.choice(["Dresses", "Knits", "Blouses", "Lounge", "Sweaters"], size=num_rows)
    }
    return pd.DataFrame(data)

# Plot Regression
def plot_regression(df_pandas):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df_pandas['Age'], y=df_pandas['Rating'], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.xlabel("Age")
    plt.ylabel("Rating")
    plt.title("Regression Line")
    st.pyplot(plt)

# Plot Clusters
def plot_clusters(df_pandas, predictions):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_pandas['Age'], y=df_pandas['Positive Feedback Count'], hue=predictions.squeeze(), palette='viridis')
    plt.xlabel("Age")
    plt.ylabel("Positive Feedback Count")
    plt.title("K-Means Clusters")
    st.pyplot(plt)

# Streamlit Title
st.title("Lab 10: Women's Clothing Reviews (Synthetic Data Version)")

# Sidebar with task selection
st.sidebar.title("Choose Analysis Type")
task = st.sidebar.radio("Select Option:", ["View Data", "Clean Data", "EDA", "Run Regression", "Run Clustering", "Run Classification"])

# Generate synthetic dataset
df_pandas = generate_synthetic_data()
spark = create_spark_session()
df = spark.createDataFrame(df_pandas)

# Perform the selected task
if task == "View Data":
    st.subheader("Synthetic Dataset Sample:")
    st.write(df_pandas.head())

elif task == "Clean Data":
    st.subheader("Cleaned Data Sample:")
    df_pandas = df_pandas.dropna()
    df = spark.createDataFrame(df_pandas)
    st.write(df.limit(50).toPandas())

elif task == "EDA":
    st.subheader("Exploratory Data Analysis")
    pdf = df.toPandas()
    fig, ax = plt.subplots()
    sns.histplot(pdf['Rating'], bins=5, ax=ax)
    st.pyplot(fig)

    st.subheader("Descriptive Statistics:")
    st.write(df.describe().toPandas())

elif task == "Run Regression":
    st.subheader("Linear Regression: Predicting Rating based on Age")
    assembler = VectorAssembler(inputCols=['Age'], outputCol='features')
    df_model = assembler.transform(df).select('features', col('Rating').alias('label'))
    lr = LinearRegression()
    model = lr.fit(df_model)
    st.write("Regression Coefficients:", model.coefficients)
    st.write("Intercept:", model.intercept)
    plot_regression(df_pandas)

elif task == "Run Clustering":
    st.subheader("K-Means Clustering")
    assembler = VectorAssembler(inputCols=['Age', 'Positive Feedback Count'], outputCol='features')
    df_model = assembler.transform(df).select('features')
    kmeans = KMeans(k=3)
    model = kmeans.fit(df_model)
    predictions = model.transform(df_model).select("prediction").toPandas()
    st.write("Cluster Centers:", model.clusterCenters())
    plot_clusters(df_pandas, predictions)

elif task == "Run Classification":
    st.subheader("Logistic Regression: Predicting Recommendation")
    assembler = VectorAssembler(inputCols=['Age', 'Rating'], outputCol='features')
    df_model = assembler.transform(df).select('features', col('Recommended IND').alias('label'))
    log_reg = LogisticRegression()
    model = log_reg.fit(df_model)
    st.write("Logistic Regression Coefficients:", model.coefficients)
    st.write("Intercept:", model.intercept)

    test_data = spark.createDataFrame([(30, 4), (50, 1)], ["Age", "Rating"])
    test_data = assembler.transform(test_data).select("features")
    predictions = model.transform(test_data).select("prediction").toPandas()
    st.write("Test Data Predictions - Recommended IND:")
    st.write(predictions)
