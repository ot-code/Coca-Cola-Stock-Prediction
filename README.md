# <h1 align="center">Coca Cola FEMSA - Prediction models</h1>

This project offers a deep dive into predictive analytics in the financial sector. It compares four prediction models — Linear Regression, ARIMA, XGBoost, and LSTM Neural Networks — to forecast the closing price of Coca-Cola FEMSA shares 📈.

# **Elements**

## **Data analysis**

The journey starts with preparing the dataset—cleaning the data, transforming dates, addressing missing values, and normalizing it. This is followed by an exploratory data analysis (EDA) phase to uncover trends, volatility, and correlations in prices and trading volumes.

## Feature engineering

New variables, such as the 5-day moving average, are created to capture short-term trends.

## **Predictive modeling**

Four different approaches are implemented and compared:
- **Linear regression**: Acts as a benchmark, offering an almost perfect fit but requiring caution to avoid overfitting.
- **ARIMA**: A time series model which, in this instance, yields high errors and a negative R² score.
- **XGBoost**: A robust and precise ensemble algorithm, well-suited for capturing non-linear relationships.
- **LSTM**: A neural network tailored for sequential data, capable of understanding complex temporal dependencies.


## **Model evaluation**

Se utilizan métricas como MAE, MSE y el coeficiente de determinación (R²) para medir el desempeño y determinar el modelo más confiable para la toma de decisiones financieras.

# **Tools and technologies**

- **Programming language**: Python
- **Development environment**: Visual Studio Code
- **Key libraries**:
    - **Pandas and NumPy for data manipulation and numerical operations.
    - **Matplotlib** and **Seaborn** for visualization.
    - **Scikit-Learn** for regression models and evaluation metrics.
    - Statsmodels for **ARIMA** analysis.
    - **XGBoost** for tree-based modeling.
    - **TensorFlow/Keras** for building and training LSTM neural networks.
- **Data source**: Historical price and trading volume data in CSV format from Investing.com, covering five years.

# **Objective and relevance**

This project seeks to identify the best predictive model to guide strategic financial decisions, minimize risks, and seize market opportunities. It also highlights the importance of model validation and fine-tuning to prevent overfitting and ensure adaptability to new data.
By blending traditional and modern machine learning techniques, this project demonstrates expertise in data analysis, predictive modeling, and visualization. It's a prime example of how data science can revolutionize decision-making in the business world! 🚀

---

<p align="center">
<big><strong>✨Thank you for visiting this repository!✨</strong></big>
</p>
