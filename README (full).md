# <h1 align="center">Coca Cola FEMSA - Prediction models</h1>

# **Executive summary**

This analysis compares four predictive models—Linear Regression, ARIMA, XGBoost, and LSTM Neural Networks—for forecasting Coca‑Cola FEMSA’s closing share price. This project seeks to evaluate and determine the optimal model based on key performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and the Coefficient of Determination (R²). While Linear Regression achieved an almost perfect fit (R² = 1.00), the risk of overfitting requires further validation. Both XGBoost and LSTM, with R² values around 0.97–0.98, provide robust performance with realistic error margins, making them more reliable for future forecasts. ARIMA underperforms (R² < 0) and is not recommended without significant improvements. These findings are critical for strategic financial decision-making.

---

# **Case description**

This document analyzes historical closing price data from Coca‑Cola FEMSA—one of Latin America’s top beverage bottlers—to compare different predictive models for forecasting future share prices. By revising our focus, the primary goal is now to evaluate these models against each other and select the best one for making accurate predictions, rather than developing a broader case study.

The study covers the following key phases:

- **Data preparation:** Format conversion, missing value handling, and consistency checks.
- **Exploratory data analysis (EDA):** Identifying trends, distributions, and volatility in the data.
- **Feature engineering:** Creating new features and normalizing data for improved model performance.
- **Predictive modeling and comparison:** Implementing and evaluating Linear Regression, ARIMA, XGBoost, and LSTM Neural Networks.
- **Model evaluation:** Using metrics (MAE, MSE, R²) to compare model performance and determine the optimal choice.
- **Business implications:** Assessing how the chosen model supports strategic financial decisions.

---

# **Why data analysis matters?**

Data analysis is at the heart of sound decision-making in finance. By understanding historical trends and rigorously comparing predictive models, we can reduce risk and enhance investment strategies. Advanced machine learning tools provide a deeper insight into market behavior, ensuring that the selected model is both robust and scalable for practical use.

---

# **Theoretical background**

Data analysis is a cornerstone for strategic business decisions. In this document, the focus is on comparing predictive models for share price forecasting. Key concepts include:

- **Descriptive analysis:** Exploring and summarizing data to understand trends and variability in closing prices.
- **Predictive analysis:** Statistical and machine learning models to forecast future values. Models considered include:
    - *Linear Regression:* For capturing linear relationships.
    - *ARIMA:* A time series approach for sequential data.
    - *XGBoost:* A tree-based ensemble method known for its prediction accuracy.
    - *LSTM Neural Networks:* Deep learning models that capture long-term dependencies.
- **Prescriptive Analysis:** Translating model outputs into actionable recommendations for financial strategy.

---

# **Methodologies**

To ensure a reliable and robust comparison, the following methods were employed:

## **Data mining and preparation**

- Extraction and cleaning of historical data from Investing.com.
- Conversion of date formats, numerical transformation of trading volume, and handling of missing values (using forward-fill methods).

## **Exploratory data analysis (EDA)**

- Overview of the dataset, including summary statistics and distribution analysis.
- Visualizations to identify trends, volatility, and correlations among variables.

## **Feature selection and engineering**

- Selection of key variables such as “Opening”, “High”, “Low”, “Volume”, and “% Change”.
- Creation of a 5-day moving average to capture short-term trends.

## **Predictive modeling**

- **Linear Regression:** Serves as a baseline model.
- **ARIMA:** Applied to capture time-based dependencies.
- **XGBoost:** An ensemble method for improved accuracy.
- **LSTM Neural Networks:** A deep learning approach tailored for sequential data.

## **Model evaluation**

Models were compared using:

- **MAE (Mean Absolute Error):** Average error magnitude.
- **MSE (Mean Squared Error):** Emphasizes larger errors.
- **R² Score (Coefficient of Determination):** Indicates the proportion of variance explained.

---

# **Tools and technologies**

The analysis leveraged a range of industry-standard tools:

- **Data source:** Historical data from Investing.com in CSV format.
- **Programming environment:** Visual Studio Code with Python libraries including:
    - *Pandas:* For data manipulation.
    - *NumPy:* For numerical operations.
    - *Matplotlib & Seaborn:* For visualization.
    - *Scikit-Learn:* For regression models and performance metrics.
    - *Statsmodels:* For ARIMA analysis.
    - *XGBoost:* For advanced tree-based modeling.
    - *TensorFlow/Keras:* For developing LSTM networks.
- **Data visualization:** Initial data organization in Excel

# **Data source and characteristics**

## **Data source**

The dataset was sourced from Investing.com, containing historical daily closing prices for Coca‑Cola FEMSA over the past five years.

## **Data characteristics**

Key variables include:

- *Date*: Recording each day.
- *Opening, high, low, closing prices*: Prices at different points in the trading day.
- *Trading volume*: Number of shares traded.
- *Percentage change*: Daily price changes expressed in percentage.

The data underwent comprehensive cleaning to ensure accuracy in analysis.

---

# **Limitations and biases**

While the dataset is robust, there are inherent limitations:

- **Data frequency:** Daily records may miss intraday variations.
- **External influences:** Macro events and corporate actions are not directly captured.
- **Single source bias:** Dependence on one data provider might introduce biases.

These limitations are considered when interpreting model performance.

---

# **Analysis process**

## **Data preparation**

Steps included:

- Loading the CSV data into a Pandas DataFrame.
- Converting the “Date” column using pd.to_datetime().
- Cleaning numeric columns such as volume and percentage change.
- Handling missing values with forward-fill.

## **Exploratory data analysis (EDA)**

The EDA phase provided an initial understanding of the data through:

- **Dataset overview:** Using df.info() to verify data types and non-null counts.
- **Descriptive statistics:** Calculating mean, median, standard deviation, and quartiles with df.describe().
- **Price distribution analysis:** Creating histograms for the “Closing”, “Opening”, “High”, and “Low” prices to detect patterns or anomalies.
- **Time trend analysis:** Plotting line charts to visualize the evolution of opening and closing prices over time.
- **Correlation analysis:** Computing and visualizing a correlation matrix to uncover relationships among numerical variables.
- **Volatility analysis:** Assessing daily price changes to identify periods of high and low volatility.
- **Trading volume distribution:**Analyzing the frequency and distribution of trading volumes.
- **Closing price trend:** Generating a clear line chart to illustrate the long-term trend in closing prices.

## **Feature engineering and train-test split**

- Creation of additional features like the 5-day moving average.
- Standardization of data using StandardScaler.
- Splitting data into training (80%) and test (20%) sets.

---

# **Model implementation**

The following predictive models were implemented:

## **Linear Regression**

- Serves as a benchmark with a very high R² (1.00).
- Caution: The near-perfect fit may indicate overfitting.

## **ARIMA**

- A time series approach that, in this case, yielded high error values (MAE: 25.22; MSE: 922.47; R²: -0.06).

## **XGBoost**

- An ensemble method with robust performance (MAE: 4.32; MSE: 26.43; R²: 0.97).

## **LSTM Neural Networks**

- Deep learning model yielding strong results (MAE: 3.05; MSE: 13.23; R²: 0.98).

---

# **Model evaluation**

Evaluating model performance is crucial for ensuring reliability. We used the following metrics:

- **Mean Absolute Error (MAE):** The average error between predicted and actual values.
- **Mean Squared Error (MSE):** The average of the squared errors, which is more sensitive to large discrepancies.
- **Coefficient of Determination (R²):** Measures how well the model explains the variance in the data; values closer to 1 indicate a stronger fit.

We also compared predicted values to actual closing prices visually to assess model performance.

---

# **Results**

The statistical analysis and predictive modeling of Coca Cola FEMSA's closing price offer valuable insights into the asset's volatility and the effectiveness of various time series forecasting models.

**Descriptive statistics**

The dataset under analysis comprises 1,302 records, allowing for an in-depth examination of the closing price trends. It includes variables such as "Date," "Close," "Open," "High," "Low," "Volume," and "% change," which together paint a comprehensive picture of market movements.

- **Average closing price:** 126.52
- **Value range:** From 77.30 to 183.01, indicating significant variability.
- **Standard deviation:** 26.79, reflecting moderate volatility.
- **Quartiles:**
    - Q1 (25th Percentile): 104.34
    - Q3 (75th Percentile): 151.72

The price distribution is skewed, with most values concentrated between the first and third quartiles, suggesting the presence of extreme events or abrupt price fluctuations.

### **Distribution analysis of Coca Cola FEMSA stock prices**

```
<div align="center">
  <img src="URL" 
  alt="Distribution analysis of Coca Cola FEMSA stock prices" width="500" height="650" />
  <div align="center">
    Calculadora sin ejecutar
  </div>
</div>
```

- **Distribution shape:**

The price distributions (closing, opening, high, and low) show two distinct peaks. The first peak, between 90 and 120 units, and a second peak, between 140 and 180 units, suggest shifts in market regimes influenced by various external factors.

- **Bias analysis:**

There is no strong bias observed, although very high prices occur less frequently.

- **Volatility and dispersion:**

The data shows significant dispersion, confirming periods of volatility. The similarity between opening and closing price distributions indicates that the daily range remains relatively stable, with occasional larger gaps.

- **Price comparisons:**

The histograms of opening and closing prices are nearly identical, as expected given their close relationship. The high and low prices exhibit minor differences that reflect typical daily oscillations.

### **Coca Cola FEMSA price trend**

```
<div align="center">
  <img src="URL" 
  alt="Coca Cola FEMSA price trend" width="500" height="650" />
  <div align="center">
    Calculadora sin ejecutar
  </div>
</div>
```

**Graph description**

- At first glance, both lines almost perfectly overlap, suggesting that opening and closing prices tend to be very similar on most trading days. This behavior is typical for large-cap, highly liquid stocks where intraday price swings are generally limited—except during periods of high volatility.

**Overall trend**

- The five-year period shows a significant downward impact at the beginning of 2020, followed by a recovery and a strong upward trend from 2021 to 2023, a period of consolidation in 2024, and then another surge as we approach 2025. It’s important to note that the steep decline in 2020 was largely due to the onset of the COVID-19 pandemic and the global uncertainty that accompanied it.

**Relationship between opening and closing prices**

When you compare the opening (blue) and closing (red) lines, you can see they are almost identical. This indicates:

- A very high correlation, meaning that on most days the closing price is nearly the same as the opening price.
- Low intraday variability, so even though there are days with higher volatility, the chart doesn’t show large, consistent daily gaps.
- A typical behavior of a stable or highly liquid stock where investors can easily adjust prices during the trading day.

**Financial and statistical interpretation**

- **Correlation Between opening and closing prices:**

It's evident from the graph that the opening and closing prices remain very close on most days, suggesting a correlation near 1. In other words, knowing the opening price can reasonably predict the range in which the closing price will fall—unless unexpected news or events occur.

- **Intraday volatility:**

Although there are moments of high volatility (especially during 2020), the difference between the opening and closing prices is usually small. This shows that the market tends to fluctuate moderately during the day, reflecting a relatively controlled intraday risk for the stock. However, during uncertain times, this gap can increase significantly.

- **Closing price prediction:**

The strong similarity between opening and closing prices suggests that simple prediction models (for example, assuming the closing price will be similar to the opening price) could serve as a good starting point. However, to improve accuracy, it’s advisable to include volatility indicators, corporate events (like quarterly earnings), and macroeconomic factors. These aspects usually account for the more extreme intraday movements that a simple analysis might miss.

**Summary**

- **High intraday correlation:** The chart clearly shows that the closing price usually stays close to the opening price, except on days with unusual news or volatility.
- **Moderate volatility:** Most days exhibit limited intraday differences, although significant drops or surges can be seen during critical periods, such as in 2020.
- **Long-term trend:** Following the sharp decline in 2020, there’s a sustained upward trend, reflecting a possible strengthening of the company in the market.
- **Implications for forecasting:** While the opening price is a key factor in predicting the closing price, monitoring additional volatility and economic indicators is essential for more accurate forecasts.
- **Overall insight:** The five-year graph of opening and closing prices illustrates a story of recovery and growth, briefly interrupted by the negative impact of 2020. For any investor or analyst, this information confirms that, under normal conditions, Coca-Cola FEMSA’s stock shows moderate intraday movements and an almost perfect correlation between opening and closing prices. However, it also highlights the importance of watching out for periods of exceptional volatility because during those times, even a slight gap between the opening and closing prices can significantly influence investment decisions and forecasting models.

### **Correlation matrix insights**

```
<div align="center">
  <img src="URL" 
  alt="Correlation matrix insights" width="500" height="650" />
  <div align="center">
    Calculadora sin ejecutar
  </div>
</div>
```

This correlation matrix allows us to identify which variables are worth including in our analysis and which might be redundant. It also shows the need to consider additional indicators—like historical volatility, corporate calendars, or macroeconomic factors—that aren’t immediately apparent in a simple correlation study.

**Matrix overview**

The correlation matrix presents values ranging from -1 to +1:

- **+1:** A perfect positive correlation—when one variable rises, the other rises proportionally.
- **0:** No clear linear relationship exists between the two variables.
- **1:** A perfect negative correlation, meaning that when one variable increases, the other decreases in exact proportion.

**Financial and statistical insights**

- **High correlation among same-day prices:**

The closing, opening, and high prices move almost in lockstep. This pattern is common in well-established, highly liquid stocks, where intraday fluctuations tend to be modest and consistent.

- **Daily low price shows lower correlation:**

The low price often acts as an outlier—perhaps due to a brief sell-off—and does not always align with the other prices, which explains its lower correlation.

- **% change has little direct link to price levels:**

The weak correlation between the percentage change and the actual prices suggests that these percentage shifts are not directly tied to the current price level.

- **Volume displays a mild negative correlation with prices:**

Trading volume doesn’t strictly follow the price. Instead, it is influenced by specific news, quarterly earnings reports, or other external factors. The moderate correlation with the percentage change indicates that on days with large price movements, increased investor activity typically leads to higher trading volumes.

**Recap**

Overall, the correlation matrix clearly illustrates how the price features (closing, opening, high, and low), daily percentage changes, and trading volume are interconnected. Generally speaking:

- The closing, opening, and high prices exhibit an almost perfect linear relationship, which points to moderate intraday volatility and synchronized price movements.
- The daily low does not always align with the other prices, suggesting occasional spikes or dips in volatility during a session.
- The percentage change does not depend solely on the price levels, while volume only shows a moderate connection with price movements—highlighting the complex nature of market participation.

### **Volatility analysis**

```
<div align="center">
  <img src="URL" 
  alt="Volatility analysis" width="500" height="650" />
  <div align="center">
    Calculadora sin ejecutar
  </div>
</div>
```

**Defining volatility**

Volatility refers to the degree of variation in an asset's price over time. This concept is crucial for investors and analysts for two main reasons:

- **Risk:** Higher volatility indicates greater risk.
- **Opportunity:** For traders, substantial price swings can create opportunities for quick gains, though they also increase the likelihood of significant losses.

**Key observations**

- **Volatility peaks around 2020:**

The beginning of 2020 witnessed a significant increase in volatility, as shown by the significant height of the bars on the chart. During this time, the COVID-19 pandemic was at its most critical phase.

- **Gradual decline through 2021-2022:**

Volatility began to return to its normal level after the initial crisis. Lower and more stable bar heights during this period can be seen on the chart.

- **Moderate fluctuations in 2023:**

In 2023, the data does not show a sustained period of high volatility; rather, it presents sporadic peaks, indicating intermittent bursts of volatility.

- **New peaks in late 2024 and 2025:**

Toward the right end of the chart, notably in late 2024 and 2025, the bars rise significantly, suggesting a recent increase in volatility. This uptick may be due to several reasons, including the controversy surrounding the company's relationship with the Latino market and migrants in 2025.

**Additional metrics for measuring volatility.**

While the chart displays daily absolute changes, volatility can be assessed using other methods. For example, we can:

- Visualize volatility on an annualized basis to gauge yearly risk.
- Analyze fluctuations within individual trading sessions.
- Consider other timeframes or statistical measures to gain a more comprehensive view of market risk.

**Outline**

- **Notable variations over time:**

Coca-Cola FEMSA's closing price volatility has seen a significant change in the past five years, particularly during the crisis in 2020 and some recent rises in 2024-2025.

- **Stable vs. turbulent periods:**

A stable market environment is indicated by periods of relative calm, while peaks are indicated by episodes of heightened uncertainty or strong reactions to major news events.

- **Implications for forecasting:**

A deep understanding of daily volatility is essential for evaluating the uncertainty in closing price forecasts. Prediction is more challenging when there are significant volatility spikes, which means that more robust models or additional data (such as news updates and earnings calendars) are needed.

### **Trading volume analysis**

```
<div align="center">
  <img src="URL" 
  alt="Trading volume analysis" width="500" height="650" />
  <div align="center">
    Calculadora sin ejecutar
  </div>
</div>
```

**Graph description**

- The horizontal axis shows the daily trading volume (the number of shares traded).
- The vertical axis indicates the number of days that fall into each volume range.
- The curve over the histogram represents the density, illustrating how trading volume frequencies are distributed over the period.

**Distribution shape**

- There is a pronounced peak at lower volume levels, indicating that most days experience relatively moderate trading activity.
- The distribution is right-skewed, meaning that on a few occasions, the trading volume spikes to levels significantly higher than the average.

**Key points**

- While most days have moderate trading volumes, there are occasional significant peaks.
- Understanding this distribution helps investors and analysts better assess liquidity and anticipate the potential impact of specific news or events on market behavior.

### **Closing price trend**

```
<div align="center">
  <img src="URL" 
  alt="Closing price trend" width="500" height="650" />
  <div align="center">
    Calculadora sin ejecutar
  </div>
</div>
```

**Graph description**

- **X-axis (Date)**: Covers the period from 2020 to 2025, with daily or frequent points showing the closing price's temporal evolution.
- **Y-axis (Closing Price)**: Ranges approximately from 80 (the lowest levels around 2020) to values exceeding 180 (by 2025).
- **Main Curve (Red Line)**: Represents the stock's closing price at the end of each market session. Visually, the curve shows ups and downs but reveals an upward trend when considering the entire period.

**Trend phases**

1. **2020 – sharp drop and slight recovery**:
    1. A notable decline is seen, with prices dropping from around 110-115 to roughly 80-85 within a relatively short span.
    2. This drop aligns with the global COVID-19 pandemic, which significantly impacted most financial markets.
    3. After this low, a rebound emerges. As markets adapted to the pandemic, moderate optimism and greater stability prevailed.
    4. Factors like the gradual reopening of the economy and adjustments in supply chains contributed to a progressive price increase.
2. **2021 to 2023 – Bullish trend**:
    1. From 2021 onward, the curve inclines steadily, surpassing pre-2020 levels.
    2. Market confidence in beverage demand stability strengthens the company’s perceived value.
    3. Potential drivers: positive financial results, improved distribution margins, and global economic recovery.
3. **2024 – Consolidation phase**:
    1. In 2024, prices enter a plateau, marked by moderate fluctuations, without the pronounced growth of 2021-2023.
4. **2025 – New bullish momentum**:
    1. Toward the end of the graph, prices resume notable growth, reaching or exceeding 180, surpassing previous peaks.

**Key factors influencing the trend**

- **Macroeconomic context**:
    - Periods of low-interest rates, fiscal or monetary stimuli, and post-crisis economic recovery likely injected capital into the stock market, boosting valuations.
    - Exchange rate fluctuations also affect companies operating across multiple countries.
- **Corporate performance**:
    - Expanding distribution networks, strategic agreements, or product innovations positively influence investors' perception of value.
- **Market sentiment and speculation**:
    - At times, stocks may rise due to widespread optimism (bull market) beyond fundamentals. This explains the accelerated gains, particularly between 2021 and 2023.
    - Similarly, corrections and consolidation periods (2024) reflect investor caution or profit-taking.

**Highlights**

- Coca-Cola FEMSA's closing price experienced a sharp initial drop in 2020, followed by a recovery and a predominantly upward period through 2025. This demonstrates the company’s resilience and the consumer sector’s adaptability in challenging scenarios.
- The 2021-2023 period stands out for consistent growth, while 2024 presents a plateau phase. In 2025, a new bullish stretch pushes prices to record levels within this timeframe.
- For those projecting the closing price, it’s vital to account for trend changes, volatility, and the influence of macroeconomic events.

In summary, the closing price trend graph underscores Coca-Cola FEMSA’s recovery capacity after challenging periods and suggests that over the five-year horizon, the stock has shown increasing returns that could persist with solid fundamentals and favorable market conditions.

### **Comparison of prediction models**

```
<div align="center">
  <img src="URL" 
  alt="Comparison of prediction models" width="500" height="650" />
  <div align="center">
    Calculadora sin ejecutar
  </div>
</div>
```

**Graph description**

1. **X-axis (time index):** Displays the chronological sequence of observations (e.g., days).
2. **Y-axis (closing price):** Reflects both the actual and predicted stock prices.
3. **Overlapping lines:**
    - **Real (black):** The true historical closing prices.
    - **Linear regression (orange dotted):** Predictions from a linear regression model (either simple or multivariable).
    - **ARIMA (blue dotted):** Forecasts using a classic time series model.
    - **XGBoost (purple dotted):** Predictions from a gradient boosting method popular for tabular data.
    - **LSTM (gray dotted):** Outputs from a recurrent neural network designed to capture time-based dependencies.

We compared four models for predicting Coca-Cola FEMSA’s closing price using the following metrics:

- **MAE (Mean Absolute Error):** The average difference between predicted and actual values (lower is better).
- **MSE (Mean Squared Error):** The average of squared differences (again, lower is better).
- **R² Score (Coefficient of Determination):** Measures how well the model’s predictions match the actual data (values close to 1 are excellent).

**Linear regression**

- **MAE:** 0.44
- **MSE:** 0.38
- **R² Score:** 1.00

*Interpretation:*

- An MAE of 0.44 indicates that, on average, the linear regression model’s predictions are less than half a point off from the actual closing prices.
- Similarly, an MSE of 0.38 suggests overall errors are quite minimal, with the average squared error remaining below 1.
- An R² of 1.00 signifies an almost perfect fit to the data. While this is theoretically ideal, it might hint at overfitting if the model is inappropriately leveraging information from the training set. However, if the validation process is correctly implemented—using truly separate test data—it suggests that the closing prices are almost entirely explained by the predictor variables over this period.

**ARIMA**

- **MAE:** 7.30
- **MSE:** 116.91
- **R² Score:** -0.24

*Interpretation:*

- An MAE of 7.30 indicates significantly larger errors compared to the other models.
- The MSE of 116.91 reveals that certain errors can be quite substantial, as the squared differences accumulate to high values.
- A negative R² (-0.24) suggests that ARIMA underperforms even a naive model, such as one that always predicts the historical average.

**XGBoost**

- **MAE:** 0.84
- **MSE:** 2.30
- **R² Score:** 0.98

*Interpretation:*

- An MAE of 0.84 is quite good, although higher than Linear Regression’s 0.44.
- The MSE of 2.30 suggests that the squared errors remain at a manageable level.
- An R² of 0.98 indicates that XGBoost explains most of the variation in the closing price. This algorithm can effectively capture non-linear relationships if enough data is available and hyperparameters are properly optimized.

**LSTM**

- **MAE:** 1.18
- **MSE:** 1.73
- **R² Score:** 0.98

*Interpretation:*

- An MAE of 1.18 indicates that the LSTM model’s predictions are, on average, off by a little over one point from the actual closing prices, which is somewhat higher than the errors of linear regression and XGBoost.
- The MSE of 1.73 suggests moderate errors, slightly lower than XGBoost’s 2.30.
- An R² of 0.98 shows that the LSTM, which specializes in time-series data, captures most of the price movements. This type of neural network excels at discovering complex temporal patterns, though it requires more training time and careful tuning, such as adjusting layers, neurons, and learning rate.

**Summary:**

**Linear regression (R² = 1.00):**

- Near-perfect fit with smallest average deviation (MAE = 0.44).
- Verify performance on separate test data to ensure no data leakage or overfitting.
- Indicates a strong relationship between predictor variables (e.g., Open, High, Low) and closing price.

**ARIMA (Negative R²):**

- Fails to capture series’ behavior, possibly due to non-stationarity or poor parameter tuning.
- Least suitable method given its poor metrics.

**XGBoost and LSTM (R² = 0.98):**

- Both models perform excellently, slightly behind linear regression in R².
- XGBoost: Lower MAE (0.84) but slightly higher MSE (2.30) than LSTM.
- LSTM: Higher MAE (1.18) but lower MSE (1.73) than XGBoost.
- Both are reliable for predicting closing prices.
- XGBoost is easier to train and interpret.
- LSTM excels at capturing complex, time-dependent patterns.

**Business applications**

- **Data-driven decision making:**

Accurate predictions (low MAE and high R²) boost confidence in trading decisions or internal investment strategies. Conversely, poor metrics indicate the need for model adjustments or more data.

- **Operational planning:**

Better forecasts help in planning debt issuance, share buybacks, or expansion projects by aligning financial strategies with market behavior.

- **Risk and scenario analysis:**

Reliable models allow us to estimate stock performance under adverse conditions—like inflation spikes or sudden drops in demand—so that we can plan proactive risk management measures.

**Graph insights**

- The near-perfect alignment of linear regression with the actual data (R² = 1.00) is impressive but should be validated for potential overfitting.
- ARIMA’s wider error margins show its inability to capture the highs and lows of the series.
- Both XGBoost and LSTM approximate the undulating patterns well, with only minor discrepancies at key turning points.

---

# **General thoughts**

This comparison highlights linear regression’s standout performance (R² = 1.00) alongside the robust capabilities of XGBoost and LSTM (R² around 0.97–0.98). In contrast, ARIMA struggles to adapt to this dataset. Before finalizing any model for practical use, especially the linear regression model, further validation is necessary to avoid overfitting or data leakage. Once validated, machine learning approaches like XGBoost and LSTM, or even a well-justified linear regression, emerge as the most robust options for predicting Coca-Cola FEMSA’s closing prices.

---

# **Discussion**

The results offer a nuanced view of the predictive models’ effectiveness:

- **Overfitting:** The linear regression, XGBoost, and LSTM models achieve an almost perfect fit during training. However, this level of precision calls for further validation to confirm that these models perform reliably on new, unseen data.
- **ARIMA performance:** The underwhelming performance of the ARIMA model suggests that it may benefit from re-specification or the incorporation of additional explanatory variables to boost its predictive accuracy.
- **Regularization:** Implementing regularization techniques, such as L1 or L2, for XGBoost and LSTM can enhance their ability to generalize to new data.

---

# **Business implications**

For a financial decision-maker at Coca‑Cola, selecting the optimal predictive model is essential:

- **Strategic decision-making:** Accurate forecasts support critical decisions such as buy/sell recommendations, debt issuance, and expansion planning.
- **Risk management:** Reliable models help in estimating potential downturns and managing investment risk.
- **Operational planning:** Enhanced prediction accuracy allows better alignment of financial strategies with market trends.

***Recommendation:*** Further validate XGBoost and LSTM through cross-validation and testing on independent datasets. While the linear regression model appears promising, its near-perfect fit warrants caution. With appropriate regularization and validation, XGBoost and LSTM emerge as the most robust options for forecasting Coca‑Cola FEMSA’s closing prices.

---

# **Final thoughts**

By rigorously evaluating model performance, the study identifies XGBoost and LSTM as the leading approaches for forecasting closing share prices, while highlighting the potential pitfalls of overfitting in simpler models like linear regression. These insights provide a strong foundation for data-driven strategic decisions within Coca‑Cola.

---

```
<p align="center">
<big><strong>✨¡Gracias por visitar este repositorio!✨</strong></big>
</p>
```