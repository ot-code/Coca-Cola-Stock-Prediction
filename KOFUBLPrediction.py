import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

warnings.filterwarnings("ignore")

# Data loading and preprocessing
def cargar_datos(file_path):
    df = pd.read_csv(file_path)

    # Convert dates
    try:
        df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d.%m.%Y")
    except:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors='coerce')

    # Convert volume by removing "M" and "K" and remove "%" from % change
    df["Vol."] = df["Vol."].replace({"M": "*1e6", "K": "*1e3"}, regex=True).map(lambda x: eval(x) if isinstance(x, str) else x)
    df["% var."] = df["% var."].replace({"%": ""}, regex=True).astype(float)

    df.ffill(inplace=True)  # Filling missing values
    return df


# Exploratory Data Analysis (EDA)
def analisis_exploratorio(df):
    print("🔹 Información general del dataset:")
    print(df.info())
    print("\n🔹 Estadísticas descriptivas:")
    print(df.describe())

    # Graph configuration
    plt.style.use("ggplot")

    # 1. Distribution of stock prices (histograms)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))  # Creating subgraphs
    columnas = ["Cierre", "Apertura", "Máximo", "Mínimo"]
    for ax, col in zip(axes.flat, columnas):
        ax.hist(df[col], bins=30, edgecolor="black")
        ax.set_title(f"Distribución de {col[0].lower()}{col[1:]}")
    plt.suptitle("Distribución de precios de las acciones", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust design to avoid overlapping
    plt.show()

    # 2. Trends over time (timeline)
    plt.figure(figsize=(12, 5))
    sns.lineplot(x=df["Fecha"], y=df["Cierre"], label="Cierre")
    sns.lineplot(x=df["Fecha"], y=df["Apertura"], label="Apertura")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.title("Tendencia de precios de Coca Cola FEMSA")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # 3. Correlation matrix (relationships between features)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[["Cierre", "Apertura", "Máximo", "Mínimo", "% var.", "Vol."]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlación entre variables")
    plt.show()

    # 4. Volatility (absolute changes in closing price)
    df["Volatilidad"] = df["Cierre"].diff().abs()
    plt.figure(figsize=(12, 5))
    sns.lineplot(x=df["Fecha"], y=df["Volatilidad"], label="Volatilidad en precio de cierre", color="red")
    plt.xlabel("Fecha")
    plt.ylabel("Cambio absoluto en cierre")
    plt.title("Análisis de volatilidad en precio de cierre")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    # 5. Trading volume distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Vol."], bins=30, kde=True, color="purple")
    plt.xlabel("Volumen de transacciones")
    plt.ylabel("Frecuencia")
    plt.title("Distribución del volumen de transacciones")
    plt.show()

    # 6. Trend visualization (Closing Price)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="Fecha", y="Cierre", label="Precio de cierre")
    plt.title("Tendencia del precio de cierre")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de cierre")
    plt.legend()
    plt.show()

# Data preparation for Machine Learning
def preparar_datos(df):
    features = ["Apertura", "Máximo", "Mínimo", "Vol.", "% var."]
    df["Media_Movil_5"] = df["Cierre"].rolling(window=5).mean()
    df.dropna(inplace=True)

    X = df[features + ["Media_Movil_5"]]
    y = df["Cierre"]

    # Train-test split WITHOUT shuffling
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=False
    )

    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Model training
def entrenar_modelos(X_train_scaled, X_test_scaled, y_train, y_test):
    modelos = {}

    # Linear regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    modelos["Regresión Lineal"] = lr_model.predict(X_test_scaled)

    # 📌 ARIMA
    arima_model = ARIMA(y_train, order=(5, 1, 0))
    arima_model_fit = arima_model.fit()
    modelos["ARIMA"] = arima_model_fit.forecast(steps=len(y_test))

    # 📌 XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    modelos["XGBoost"] = xgb_model.predict(X_test_scaled)

    return modelos

# LSTM Implementation
def entrenar_lstm(X_train_scaled, X_test_scaled, y_train, y_test):
    scaler_lstm = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler_lstm.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_lstm.transform(y_test.values.reshape(-1, 1))

    X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    # LSTM Neural Network Architecture
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=25),
        Dense(units=1)
    ])

    # Compilation and training
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_reshaped, y_train_scaled, epochs=50, batch_size=16, verbose=1)

    # Predictions
    y_pred_lstm_scaled = model.predict(X_test_reshaped)
    y_pred_lstm = scaler_lstm.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1))

    return y_pred_lstm.flatten()

# Model evaluation and  visualization
def evaluar_modelos(modelos, y_test):
    for name, y_pred in modelos.items():
        print(f"\n📊 {name}:")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
        print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

    # Model comparison graph
    plt.figure(figsize=(12, 6))

    # Test indices
    indices_test = np.arange(len(y_test))

    # Plot the real values
    plt.plot(indices_test, y_test.values, label="Real", color="black", linewidth=2)

    # Plot the predictions of each model
    if "Regresión Lineal" in modelos:
        plt.plot(indices_test, modelos["Regresión Lineal"], label="Regresión lineal", linestyle="dashed", alpha=0.8)
    if "ARIMA" in modelos:
        plt.plot(indices_test, modelos["ARIMA"], label="ARIMA", linestyle="dotted", alpha=0.8)
    if "XGBoost" in modelos:
        plt.plot(indices_test, modelos["XGBoost"], label="XGBoost", linestyle="dashdot", alpha=0.8)
    if "LSTM" in modelos:
        plt.plot(indices_test, modelos["LSTM"], label="LSTM", linestyle="solid", alpha=0.8)

    # Visual enhancements
    plt.legend()
    plt.title("Comparación de modelos de predicción")
    plt.xlabel("Índice de tiempo")
    plt.ylabel("Precio de cierre")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# Pipeline execution
if __name__ == "__main__":
    # CSV file path
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the script's path
    file_path = os.path.join(script_dir, "KOFUBL-2.csv") # Use the same folder as the script

    # Load and analyze data
    df = cargar_datos(file_path)
    analisis_exploratorio(df)

    # Data preparation
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preparar_datos(df)

    # Training of traditional models
    modelos = entrenar_modelos(X_train_scaled, X_test_scaled, y_train, y_test)

    # LSTM training
    modelos["LSTM"] = entrenar_lstm(X_train_scaled, X_test_scaled, y_train, y_test)

    # Evaluation and visualization
    evaluar_modelos(modelos, y_test)
