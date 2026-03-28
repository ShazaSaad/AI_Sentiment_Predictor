# AI Stock Sentiment & Forecasting Dashboard

## Project Overview

This project develops an **AI-based stock forecasting and sentiment analysis system**. The system has two main independent components:

1. **Sentiment Analysis Pipeline**: Automatically collects financial news, performs sentiment analysis using Natural Language Processing (NLP), and aggregates daily sentiment scores.
2. **Stock Forecasting Pipeline**: Uses time-series forecasting models to predict stock price movements based on historical price data.

The forecasting pipeline uses **Prophet time-series forecasting** to analyze historical stock price data and generate predictions for future market trends, evaluated using directional accuracy, RMSE, and R² metrics.

The goal of this project is to provide **comprehensive stock analysis through both sentiment indicators and technical price forecasting**.

---

# System Architecture

The project consists of two independent analysis components that can be run separately or together:

## 1. Sentiment Analysis Pipeline

```text
Financial News API (Benzinga)
        │
        ▼
News Collection
(fetch_benzinga_news.py)
        │
        ▼
News Preprocessing
(clean_news.py)
        │
        ▼
Sentiment Analysis
(sentiment_model.py)
        │
        ▼
Daily Sentiment Aggregation
(sentiment_pipeline.py)
        │
        ▼
Sentiment Dataset
(daily_sentiment_scores.csv)
```

## 2. Stock Forecasting Pipeline

```text
Historical Stock Price Data (yfinance)
        │
        ▼
Price Data Loading
(load_price_data)
        │
        ▼
Prophet Time-Series Forecasting
(forecast_prices)
        │
        ▼
Forecast Evaluation
(evaluate_forecast - Directional Accuracy, RMSE, R²)
        │
        ▼
Visualization
(plot_forecast, plot_yearly_forecasts)
        │
        ▼
Multi-Ticker Batch Processing (25 stocks)
(run_forecast_for_tickers)
        │
        ▼
Evaluation Results & Plots
(evaluation_results.csv, /plots/)
```

## 3. Interactive Dashboard

Both components feed into a unified dashboard for visualization and analysis.

---

# Project Structure

```text
AI_Sentiment_Predictor
│
├── forecasting_pipeline/
│   ├── plots/
│   ├── evaluation_results.csv
│   ├── forecasting_model.ipynb
│   └── results.ipynb
│
├── src/
│   ├── data/
│   │   ├── raw_news/
│   │   │   └── benzinga_news_raw.csv
│   │   │
│   │   ├── processed_news/
│   │   │   └── cleaned_news.csv
│   │   │
│   │   └── sentiment/
│   │       └── daily_sentiment_scores.csv
│   │
│   ├── data_collection/
│   │   └── fetch_benzinga_news.py
│   │
│   ├── preprocessing/
│   │   └── clean_news.py
│   │
│   ├── sentiment/
│   │   ├── sentiment_model.py
│   │   └── sentiment_pipeline.py
│   │
│   └── main_sentiment_pipeline.py
│
├── app.py
└── README.md
```

---

# Sentiment Analysis Pipeline

The sentiment pipeline consists of four main stages.

---

## 1. Financial News Collection

Script:

```text
src/data_collection/fetch_benzinga_news.py
```

This script retrieves financial news using the **Benzinga News API** and stores it as raw data.

Output:

```text
src/data/raw_news/benzinga_news_raw.csv
```

The collected dataset typically contains:

- news title
- article content
- publication date
- URL
- associated ticker symbols

---

## 2. News Preprocessing

Script:

```text
src/preprocessing/clean_news.py
```

Preprocessing tasks include:

- removing HTML tags
- removing duplicate articles
- handling missing values
- normalizing text formatting

Output:

```text
src/data/processed_news/cleaned_news.csv
```

This dataset contains cleaned news text ready for sentiment analysis.

---

## 3. Sentiment Analysis

Script:

```text
src/sentiment/sentiment_model.py
```

This component analyzes financial news text and generates a **sentiment score** for each article.

The sentiment classification determines whether the news is:

- Positive
- Neutral
- Negative

---

## 4. Daily Sentiment Aggregation

Script:

```text
src/sentiment/sentiment_pipeline.py
```

Sentiment scores are aggregated by date to generate **daily sentiment indicators**.

Output file:

```text
src/data/sentiment/daily_sentiment_scores.csv
```

These indicators represent the overall market sentiment extracted from financial news.

---

# Forecasting Pipeline

## Overview

The forecasting pipeline analyzes **historical stock price data** using the **Prophet time-series forecasting model** to generate price predictions for 25 major stocks (AAPL, MSFT, NVDA, AMZN, GOOG, META, TSLA, BRK-B, JPM, JNJ, V, UNH, PG, MA, DIS, HD, PFE, BAC, NFLX, ORCL, CMCSA, XOM, CVX, UNP, KO).

Location:

```text
forecasting_pipeline/
```

## Key Features

- **Prophet-based Forecasting**: Uses Facebook's Prophet library for robust time-series analysis
- **Multi-Ticker Support**: Analyzes 25 major stocks in a single batch run
- **Comprehensive Visualization**: Generates historical, future, and yearly forecast plots
- **Multiple Metrics**: Evaluates forecasts using Directional Accuracy, RMSE, and R²

## Notebook Files

- `forecasting_model.ipynb` - Main forecasting notebook with multi-ticker pipeline
- `results.ipynb` - Results analysis notebook

## Output Files

Evaluation metrics are stored in:

```text
forecasting_pipeline/evaluation_results.csv
```

Generated plots are organized in:

```text
forecasting_pipeline/plots/
├── historical/     (historical price and model fit visualizations)
├── future/         (future price forecast visualizations)
└── yearly/         (yearly breakdown plots)
```

## Evaluation Metrics

The forecasting models are evaluated using:

| Metric               | Description                                          |
| -------------------- | ---------------------------------------------------- |
| Directional Accuracy | % of time model correctly predicts price direction   |
| RMSE                 | Root Mean Squared Error - prediction error magnitude |
| R²                   | Coefficient of determination - goodness of fit       |

---

# Dashboard

The project includes an interactive dashboard built with **Streamlit**.

Dashboard features:

- company selection
- forecasting visualizations
- model evaluation metrics
- sentiment trend analysis
- recent financial news

Main dashboard file:

```text
app.py
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ShazaSaad/AI_Sentiment_Predictor.git
```

Navigate to the project directory:

```bash
cd AI_Sentiment_Predictor
```

Install required dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn requests nltk beautifulsoup4 transformers torch prophet yfinance
```

---

# Running the Sentiment Pipeline

Generate sentiment data by running the pipeline:

```bash
cd src
python main_sentiment_pipeline.py
```

This process will create:

```text
src/data/sentiment/daily_sentiment_scores.csv
```

---

# Running the Forecasting Pipeline

The forecasting pipeline is included in a Jupyter notebook. To run the complete forecast for 25 stocks:

1. Open the notebook:

   ```text
   forecasting_pipeline/forecasting_model.ipynb
   ```

2. Execute all cells in order. The notebook will:
   - Load historical price data for 25 major stocks
   - Train Prophet models for each stock
   - Generate forecasts and evaluate them
   - Create visualization plots (historical, future, yearly)
   - Save results to `evaluation_results.csv` and organized plot folders

3. Generated outputs:
   - Evaluation metrics: `forecasting_pipeline/evaluation_results.csv`
   - Plots: `forecasting_pipeline/plots/` (organized by type: historical/, future/, yearly/)

**Note**: The forecasting pipeline runs independently from the sentiment analysis. It uses historical stock price data only.

---

# Running the Dashboard

From the project root directory:

```bash
streamlit run app.py
```

The dashboard will be available at:

```
http://localhost:8501
```

---

# Data Flow Diagram

The system contains two independent pipelines that provide market analysis from different perspectives:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                      SENTIMENT ANALYSIS PIPELINE                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│ │ Benzinga News API│──▶│ Raw News Dataset │──▶│ News Preprocessing
│ └──────────────────┘   └──────────────────┘   └────────┬─────────┘    │
│                                                        │                 │
│                                    ┌───────────────────▼──────────┐    │
│                                    │ Sentiment Classification     │    │
│                                    │ (Positive/Neutral/Negative)  │    │
│                                    └───────────────────┬──────────┘    │
│                                                        │                 │
│                                    ┌───────────────────▼──────────┐    │
│                                    │ Daily Sentiment Aggregation  │    │
│                                    └───────────────────┬──────────┘    │
│                                                        │                 │
│                                 ┌──────────────────────▼────────┐      │
│                                 │ daily_sentiment_scores.csv    │      │
│                                 └───────────────────────────────┘      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                    STOCK FORECASTING PIPELINE                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ ┌──────────────────┐   ┌──────────────────┐   ┌───────────────────┐   │
│ │ Yahoo Finance    │──▶│ Historical Price │──▶│ Prophet Time-Series
│ │ (25 Stocks)      │   │ Data (yfinance)  │   │     Forecasting    │   │
│ └──────────────────┘   └──────────────────┘   └────────┬──────────┘   │
│                                                         │                │
│                             ┌───────────────────────────▼─────────┐   │
│                             │ Forecast Evaluation                 │   │
│                             │ (Directional Accuracy, RMSE, R²)    │   │
│                             └───────────────────────────┬─────────┘   │
│                                                         │                │
│          ┌──────────────────┬──────────────────────────┼──────────┐   │
│          │                  │                          │          │   │
│      ┌───▼────┐    ┌────────▼────┐    ┌───────────────▼────┐  ┌──▼──┐
│      │Historical│   │ Forecast     │    │ Yearly Breakdowns  │  │ CSV  │
│      │    Plots  │   │  Plots       │    │    (per-year)      │  │Results
│      └──────────┘   └─────────────┘    └────────────────┘  └──────┘
│
└──────────────────────────────────────────────────────────────────────────┘

Both pipelines ──────▶ Interactive Dashboard (Streamlit) ──────▶ Visualization
                                (app.py)
```

This architecture allows sentiment analysis and price forecasting to be analyzed independently or together for comprehensive market insights.

---

# AI Model Explanation

## Sentiment Analysis Model

The sentiment analysis component analyzes financial news headlines and classifies their sentiment.

The model processes text input and predicts whether the sentiment is:

- Positive
- Neutral
- Negative

Example:

| News Title                                      | Sentiment |
| ----------------------------------------------- | --------- |
| Tesla shares surge after strong earnings report | Positive  |
| Apple faces supply chain disruptions            | Negative  |
| Microsoft announces new AI partnership          | Neutral   |

Each article receives a **sentiment score**, which is then aggregated to produce **daily sentiment trends**.

---

## Forecasting Models

The forecasting pipeline uses **Prophet (Facebook's time-series forecasting library)** to analyze historical stock price data and generate future predictions.

### Prophet Model

**Prophet** is a time-series forecasting model that:

- Handles seasonality and trends in stock price data
- Manages missing data and outliers gracefully
- Provides confidence intervals for predictions
- Works well with multiple stocks without retraining

### Model Outputs

For each stock, the model provides:

- **Historical fit**: How well the model captures past price movements
- **Future forecast**: 30-day price predictions with 80% confidence intervals
- **Yearly breakdown**: Separate visualizations for each year in the dataset
- **Performance metrics**: Directional Accuracy, RMSE, and R² scores

Example evaluation for a stock:

| Metric               | Value | Interpretation                                 |
| -------------------- | ----- | ---------------------------------------------- |
| Directional Accuracy | 68%   | Model correctly predicts direction 68% of time |
| RMSE                 | 12.5  | Average prediction error of $12.50             |
| R²                   | 0.94  | Model explains 94% of price variance           |

### Data Processing

1. **Raw Price Data**: Retrieved from Yahoo Finance (yfinance)
2. **Preparation**: Data cleaned and formatted for Prophet
3. **Training**: Prophet model trained on available historical data
4. **Validation**: Metrics calculated by comparing predictions to actual prices
5. **Visualization**: Multiple plot types generated for analysis

These results are stored in:

```text
forecasting_pipeline/evaluation_results.csv
```

---

# Example Dashboard Output

The dashboard provides an interactive interface for exploring the results of both the sentiment analysis and forecasting pipelines.

**Dashboard features:**

- Selecting different companies
- Viewing stock price forecasting graphs
- Analyzing historical vs. predicted prices
- Exploring sentiment trends (from sentiment analysis pipeline)
- Viewing forecast evaluation metrics
- Recent financial news

The dashboard integrates both independent pipelines to provide a comprehensive view of stock analysis.

---

# Technologies Used

**Core Libraries:**

- Python
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Matplotlib & Seaborn - Data visualization
- Scikit-learn - Machine learning metrics

**Forecasting:**

- Prophet (Facebook) - Time-series forecasting
- yfinance - Yahoo Finance data retrieval

**Sentiment Analysis:**

- Natural Language Processing (NLP)
- Transformers library - Pre-trained language models
- Financial News APIs (Benzinga)
- BeautifulSoup4 - Web scraping

**Dashboard & Web:**

- Streamlit - Interactive web dashboard
- Requests - HTTP library

---

# Future Improvements

Potential improvements include:

- **Sentiment-Price Integration**: Combine sentiment analysis with forecasting models to improve predictions
- **Advanced Forecasting Models**: Explore LSTM, ARIMA, and ensemble methods
- **Real-time Data Streaming**: Implement live price and news updates
- **Social Media Sentiment**: Integrate sentiment from Twitter, Reddit, and other social platforms
- **Portfolio Optimization**: Build portfolio allocation strategies based on multi-stock analysis
- **Automated Trading Signals**: Generate actionable trading signals from combined analysis
- **Risk Analysis**: Calculate Value at Risk (VaR) and other risk metrics
- **More Stocks**: Expand forecasting to include additional stocks and sectors

---

# Authors

DSAI 4201 Project
AI Stock Sentiment & Forecasting System

Shaza Saad - 60301815
Sara Mohamed - 60101453
