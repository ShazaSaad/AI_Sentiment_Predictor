from __future__ import annotations

import ast
import math
import os
import sys
from typing import Dict, Optional

import pandas as pd
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from the converted news_model module.
# Assumes news_model.py lives in a sibling "sentiment" package, i.e.:
#   <project_root>/
#       sentiment/
#           news_model.py
#       test.py
sys.path.append(os.path.abspath(".."))
import sentiment.news_model as news_model


# ─── Price data ──────────────────────────────────────────────────────────────

def load_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load historical price data for a given ticker symbol from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the historical data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the historical price data with columns
                      'Date', 'Close', and 'Industry'.
    """
    data = yf.download(ticker, start=start_date, end=end_date)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)

    ticker_obj = yf.Ticker(ticker)
    industry = ticker_obj.info.get("industry", "Unknown")

    data["Industry"] = industry

    return data[["Date", "Close", "Industry"]]


def display_price_data(data: pd.DataFrame, ticker: str) -> None:
    """
    Display the historical price data using a line plot.

    Args:
        data (pd.DataFrame): A DataFrame containing the historical price data
                             with columns ['Date', 'Close'].
        ticker (str): The stock ticker symbol for labeling the plot.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Date", y="Close", data=data)
    plt.title(f"Historical Closing Prices for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.grid()
    plt.show()


# ─── News data ────────────────────────────────────────────────────────────────

def load_news_data() -> pd.DataFrame:
    """
    Load news data from MongoDB via news_model.

    Returns:
        pd.DataFrame: A DataFrame containing the news data.
    """
    news_df = news_model.load_news_dataframe()
    if news_df.empty:
        raise RuntimeError("News data is required for forecasting but no news was found in MongoDB.")
    return news_df


# ─── Forecasting ──────────────────────────────────────────────────────────────

def forecast_prices(data: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Forecast future stock prices using the Prophet model.

    Args:
        data (pd.DataFrame): A DataFrame containing the historical price data
                             with columns ['Date', 'Close', 'Industry'].
        periods (int): The number of future periods (days) to forecast.

    Returns:
        pd.DataFrame: A DataFrame containing the forecasted prices with columns
                      ['ds', 'yhat', 'yhat_lower', 'yhat_upper'].
    """
    df = data.rename(columns={"Date": "ds", "Close": "y", "Industry": "industry"})
    df = df.dropna(subset=["ds", "y"]).copy()

    if df.empty:
        raise ValueError("No historical price data available for forecasting.")

    def _parse_news_vector(raw_vector) -> Optional[ast.List[float]]:
        if raw_vector is None:
            return None
        if isinstance(raw_vector, str):
            try:
                raw_vector = ast.literal_eval(raw_vector)
            except (SyntaxError, ValueError):
                return None
        if isinstance(raw_vector, (list, tuple)):
            parsed = []
            for value in raw_vector:
                try:
                    parsed.append(float(value))
                except (TypeError, ValueError):
                    return None
            return parsed
        return None
    news_df = load_news_data()

    # Merge news data with price data based on assigned industry.
    # news_df is expected to have a column 'assigned_industries' containing a list
    # of industries that correspond to the 'industry' column in df.
    if not news_df.empty and "assigned_industries" in news_df.columns and "body_vector" in news_df.columns:
        for industry in df["industry"].dropna().unique():
            industry_news = news_df[
                news_df["assigned_industries"].apply(
                    lambda x: isinstance(x, (list, tuple, set)) and industry in x
                )
            ]
            if industry_news.empty:
                continue

            parsed_vectors = industry_news["body_vector"].apply(_parse_news_vector).dropna().tolist()
            if not parsed_vectors and "body" in industry_news.columns:
                fallback_bodies = industry_news["body"].fillna("").astype(str).str.strip()
                fallback_bodies = fallback_bodies[fallback_bodies != ""]
                if not fallback_bodies.empty:
                    tfidf_matrix = TfidfVectorizer(stop_words="english").fit_transform(fallback_bodies.tolist())
                    parsed_vectors = [row.toarray().flatten().tolist() for row in tfidf_matrix]

            if not parsed_vectors:
                continue

            vector_df = pd.DataFrame(parsed_vectors).dropna(axis=1, how="any")
            if vector_df.empty:
                continue

            avg_vector = vector_df.mean(axis=0).tolist()
            for i, value in enumerate(avg_vector):
                df[f"news_vector_{i}"] = value

    # Initialize and fit the Prophet model.
    model = Prophet()
    regressors = [col for col in df.columns if col.startswith("news_vector_")]
    for regressor in regressors:
        model.add_regressor(regressor)
    model.fit(df)

    # Create a DataFrame for future dates and forecast.
    future = model.make_future_dataframe(periods=periods)
    for regressor in regressors:
        future[regressor] = df[regressor].iloc[-1] if regressor in df else 0.0
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_forecast(data: pd.DataFrame, forecast: pd.DataFrame, ticker: str) -> None:
    """
    Plot historical data, model fit, and future forecast.
    """
    plt.figure(figsize=(12, 6))

    last_date = data["Date"].max()

    forecast_history = forecast[forecast["ds"] <= last_date]
    forecast_future = forecast[forecast["ds"] > last_date]

    plt.plot(data["Date"], data["Close"], label="Historical Data", color="black")

    plt.plot(
        forecast_history["ds"],
        forecast_history["yhat"],
        label="Model Fit",
        linestyle="--",
        color="blue",
    )

    plt.plot(
        forecast_future["ds"],
        forecast_future["yhat"],
        label="Forecast (Future)",
        linestyle="--",
        color="red",
    )

    plt.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="gray",
        alpha=0.2,
    )

    plt.axvline(x=last_date, color="gray", linestyle="-", label="Forecast Start")

    plt.title(f"{ticker} Stock Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ─── Evaluation ────────────────────────────────────────────────────────────

def evaluate_forecast(data: pd.DataFrame, forecast: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate the forecasted prices against actual historical prices using MAE and RMSE.

    Args:
        data (pd.DataFrame): A DataFrame containing the historical price data
                             with columns ['Date', 'Close'].
        forecast (pd.DataFrame): A DataFrame containing the forecasted prices
                                 with columns ['ds', 'yhat'].

    Returns:
        Dict[str, float]: A dictionary containing the MAE, RMSE, and R² values.
    """
    merged = pd.merge(data, forecast, left_on="Date", right_on="ds", how="inner")

    if merged.empty:
        return {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}
    
    mae = mean_absolute_error(merged["Close"], merged["yhat"])
    rmse = math.sqrt(mean_squared_error(merged["Close"], merged["yhat"]))
    r2 = 1 - (
        sum((merged["Close"] - merged["yhat"]) ** 2)
        / sum((merged["Close"] - merged["Close"].mean()) ** 2)
    )

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    forecast_periods = 30  # Forecast for the next 30 days

    # Load and display price data
    price_data = load_price_data(ticker, start_date, end_date)
    display_price_data(price_data, ticker)

    # Forecast future prices
    forecasted_prices = forecast_prices(price_data, forecast_periods)

    # Plot the historical and forecasted prices
    plot_forecast(price_data, forecasted_prices, ticker)

    # Evaluate the forecast
    evaluation_results = evaluate_forecast(price_data, forecasted_prices)

    print(f"Evaluation Results for {ticker} Forecast:")
    print(f"Mean Absolute Error (MAE): {evaluation_results['MAE']:.2f}")
    print(f"Root Mean Squared Error (RMSE): {evaluation_results['RMSE']:.2f}")
    print(f"R-squared (R2): {evaluation_results['R2']:.4f}")