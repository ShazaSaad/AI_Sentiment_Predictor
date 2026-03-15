# AI Stock Sentiment & Forecasting Dashboard

## Project Overview

This project develops an **AI-based stock forecasting and sentiment analysis system** that integrates **financial news sentiment with stock price prediction**.

The system automatically collects financial news, performs sentiment analysis using Natural Language Processing (NLP), aggregates daily sentiment scores, and visualizes the results alongside stock forecasting outputs in an interactive dashboard.

The goal of this project is to demonstrate how **market sentiment extracted from news data can support stock trend analysis and decision-making**.

---

# System Architecture

The project consists of three main components:

1. **News Data Collection**
2. **Sentiment Analysis Pipeline**
3. **Stock Forecasting & Dashboard Visualization**

Pipeline workflow:

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
        │
        ▼
Stock Forecasting Results
(forecasting_pipeline)
        │
        ▼
Interactive Dashboard
(app.py)
```

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

* news title
* article content
* publication date
* URL
* associated ticker symbols

---

## 2. News Preprocessing

Script:

```text
src/preprocessing/clean_news.py
```

Preprocessing tasks include:

* removing HTML tags
* removing duplicate articles
* handling missing values
* normalizing text formatting

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

* Positive
* Neutral
* Negative

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

The forecasting pipeline contains notebooks used to train and evaluate stock prediction models.

Location:

```text
forecasting_pipeline/
```

Contents include:

* model development notebooks
* forecasting experiments
* evaluation results

Evaluation metrics are stored in:

```text
forecasting_pipeline/evaluation_results.csv
```

---

# Dashboard

The project includes an interactive dashboard built with **Streamlit**.

Dashboard features:

* company selection
* forecasting visualizations
* model evaluation metrics
* sentiment trend analysis
* recent financial news

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
pip install streamlit pandas numpy requests nltk beautifulsoup4 transformers torch
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

The following diagram shows how information flows through the system from news collection to visualization.

```text
        ┌───────────────────────┐
        │  Benzinga News API    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ fetch_benzinga_news.py      │
        │ News Data Collection        │
        └───────────┬─────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ benzinga_news_raw.csv       │
        │ Raw News Dataset            │
        └───────────┬─────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ clean_news.py               │
        │ Text Cleaning & Filtering   │
        └───────────┬─────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ cleaned_news.csv            │
        │ Processed News Dataset      │
        └───────────┬─────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ sentiment_model.py          │
        │ Sentiment Prediction        │
        └───────────┬─────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ sentiment_pipeline.py       │
        │ Daily Sentiment Aggregation │
        └───────────┬─────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ daily_sentiment_scores.csv  │
        │ Market Sentiment Indicators │
        └───────────┬─────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ Forecasting Pipeline        │
        │ Stock Prediction Models     │
        └───────────┬─────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ Streamlit Dashboard         │
        │ app.py                      │
        └─────────────────────────────┘
```

This pipeline transforms **unstructured financial news into structured sentiment indicators** that can be used to analyze stock market behavior.

---

# AI Model Explanation

## Sentiment Analysis Model

The sentiment analysis component analyzes financial news headlines and classifies their sentiment.

The model processes text input and predicts whether the sentiment is:

* Positive
* Neutral
* Negative

Example:

| News Title                                      | Sentiment |
| ----------------------------------------------- | --------- |
| Tesla shares surge after strong earnings report | Positive  |
| Apple faces supply chain disruptions            | Negative  |
| Microsoft announces new AI partnership          | Neutral   |

Each article receives a **sentiment score**, which is then aggregated to produce **daily sentiment trends**.

---

## Forecasting Models

The forecasting pipeline analyzes historical stock price data to predict future market trends.

The models generate:

* historical price predictions
* future price forecasts (3-month horizon)
* evaluation metrics

Evaluation metrics include:

| Metric               | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| R²                   | Measures how well predictions match actual values               |
| RMSE                 | Measures prediction error magnitude                             |
| Directional Accuracy | Measures how often the model predicts price direction correctly |

These results are stored in:

```text
forecasting_pipeline/evaluation_results.csv
```

---

# Example Dashboard Output

The dashboard provides an interactive interface for exploring the results of the sentiment and forecasting pipelines.

Features include:

* selecting different companies
* viewing forecasting graphs
* analyzing sentiment trends
* exploring recent financial news

---

# Technologies Used

* Python
* Pandas
* Natural Language Processing (NLP)
* Financial News APIs
* Machine Learning Forecasting Models
* Streamlit Dashboard

---

# Future Improvements

Potential improvements include:

* integrating social media sentiment sources (Twitter / Reddit)
* real-time news data streaming
* more advanced forecasting models
* portfolio optimization strategies
* automated trading signal generation

---

# Authors

DSAI 4201 Project
AI Stock Sentiment & Forecasting System

Shaza Saad - 60301815
Sara Mohamed - 60101453