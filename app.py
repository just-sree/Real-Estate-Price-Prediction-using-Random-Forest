import streamlit as st
import pandas as pd
from src.data_processing import load_data
from src.preprocessing import preprocess_data
from src.data_split import split_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

st.title("🏡 Real Estate Price Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Data Preview", df.head())

    # Preprocess the data
    df = preprocess_data(df)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df, "price")

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    mae, mse = evaluate_model(model, X_test, y_test)
    st.write("✅ Model Evaluation:")
    st.write(f"📉 MAE: {mae}")
    st.write(f"📉 MSE: {mse}")
