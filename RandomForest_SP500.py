import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Load S&P 500 data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
del sp500['Dividends']
del sp500['Stock Splits']
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

# Create model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=777)
horizons = [2, 5, 10, 20, 40, 80, 160, 320]
predictors = []

# Create features
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_col = f"Close_Ratio_{horizon}"
    sp500[ratio_col] = sp500["Close"] / rolling_averages["Close"]
    trend_col = f"Trend_{horizon}"
    sp500[trend_col] = sp500.shift(1).rolling(horizon).sum()["Target"]
    predictors += [ratio_col, trend_col]
    
# Drop missing values
sp500 = sp500.dropna()

# Train model on current data, then predict on future data
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds = (preds > 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Iteratively train and test model on historical data
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:i+step].copy()
        preds = predict(train, test, predictors, model)
        all_predictions.append(preds)
    return pd.concat(all_predictions)

# Evaluate model precision
predictions = backtest(sp500, model, predictors)
precision = precision_score(predictions["Target"], predictions["Predictions"])
print(f"Precision: {precision:.2f}")

