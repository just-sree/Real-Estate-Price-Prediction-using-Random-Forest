import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing import load_data
from src.preprocessing import preprocess_data
from src.data_split import split_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

st.title("🏡 Real Estate Price Prediction & Visualizations")

# Load dataset from the repo (final.csv)
df = load_data("final.csv")
if df is None:
    st.error("Error loading dataset.")
else:
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Preprocess the data for modeling
    df_processed = preprocess_data(df.copy())
    X_train, X_test, y_train, y_test = split_data(df_processed, "price")  # note: using lowercase "price"
    model = train_model(X_train, y_train)
    mae, mse = evaluate_model(model, X_test, y_test)

    st.write("### Model Evaluation")
    st.write(f"**MAE:** {mae}")
    st.write(f"**MSE:** {mse}")

    st.write("### Visualizations")

    # Visualization 1: Distribution of 'price'
    st.write("#### Distribution of Price")
    plt.figure()
    plt.hist(df["price"], bins=30, edgecolor='black')
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.title("Distribution of Price")
    st.pyplot(plt.gcf())

    # Visualization 2: Scatter Plot between 'sqft' and 'price'
    st.write("#### Scatter Plot: Sqft vs Price")
    plt.figure()
    plt.scatter(df["sqft"], df["price"], alpha=0.5)
    plt.xlabel("Sqft")
    plt.ylabel("Price")
    plt.title("Sqft vs Price")
    st.pyplot(plt.gcf())

    # Visualization 3: Correlation Heatmap
    st.write("#### Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    im = plt.imshow(corr, cmap='viridis', interpolation='none')
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    st.pyplot(plt.gcf())
