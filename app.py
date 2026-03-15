import streamlit as st
import pandas as pd
import os
from PIL import Image

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="AI Stock Forecasting Dashboard",
    layout="wide"
)

st.title("AI Stock Forecasting & Sentiment Analysis")

# ----------------------------
# COMPANY LIST
# ----------------------------

companies = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "Alphabet (GOOG)": "GOOG",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA"
}

# ----------------------------
# PATHS
# ----------------------------

RESULTS_PATH = "forecasting_pipeline/evaluation_results.csv"

YEARLY_PLOTS = "forecasting_pipeline/plots/yearly"
HISTORICAL_PLOTS = "forecasting_pipeline/plots/historical"
FUTURE_PLOTS = "forecasting_pipeline/plots/future"

SENTIMENT_PATH = "src/data/sentiment/daily_sentiment_scores.csv"
NEWS_PATH = "src/data/raw_news/benzinga_news_raw.csv"

# ----------------------------
# DATA LOADING
# ----------------------------

@st.cache_data
def load_metrics():
    return pd.read_csv(RESULTS_PATH)

results_df = load_metrics()


@st.cache_data
def load_sentiment():
    if os.path.exists(SENTIMENT_PATH):
        df = pd.read_csv(SENTIMENT_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_news():
    if os.path.exists(NEWS_PATH):
        return pd.read_csv(NEWS_PATH)
    return pd.DataFrame()


sentiment_df = load_sentiment()
news_df = load_news()

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.header("Select Company")

company_name = st.sidebar.selectbox(
    "Company",
    list(companies.keys())
)

ticker = companies[company_name]

st.sidebar.write("Ticker:", ticker)

# ----------------------------
# METRICS FUNCTION
# ----------------------------

def show_metrics():

    st.subheader("Model Evaluation")

    company_results = results_df[
        results_df["ticker"] == ticker
    ]

    if company_results.empty:
        st.warning("No evaluation results found.")
        return

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average R²",
        round(company_results["R2"].mean(), 3)
    )

    col2.metric(
        "Average RMSE",
        round(company_results["RMSE"].mean(), 3)
    )

    col3.metric(
        "Directional Accuracy",
        round(company_results["Directional_Accuracy"].mean(), 3)
    )

    with st.expander("View Full Metrics Table"):
        st.dataframe(company_results)

# ----------------------------
# FORECASTING PLOTS
# ----------------------------

def show_historical_plot():

    st.subheader("Historical Performance + Prediction")

    file_path = f"{HISTORICAL_PLOTS}/{ticker}_history.png"

    if os.path.exists(file_path):
        st.image(file_path)
    else:
        st.info("Historical plot not available.")


def show_yearly_plots():

    st.subheader("Yearly Forecasts (Jan–Mar)")

    if not os.path.exists(YEARLY_PLOTS):
        st.warning("Yearly plots folder not found.")
        return

    files = sorted(os.listdir(YEARLY_PLOTS))

    found = False

    for file in files:
        if file.startswith(ticker):

            img = Image.open(os.path.join(YEARLY_PLOTS, file))

            st.image(img, caption=file)

            found = True

    if not found:
        st.info("No yearly forecasts available.")


def show_future_plot():

    st.subheader("Next 3 Month Forecast")

    file_path = f"{FUTURE_PLOTS}/{ticker}_future.png"

    if os.path.exists(file_path):
        st.image(file_path)
    else:
        st.info("Future forecast not available.")

# ----------------------------
# SENTIMENT FUNCTIONS
# ----------------------------

def show_sentiment_trend():

    st.subheader("Market News Sentiment")

    if sentiment_df.empty:
        st.warning("Sentiment data not found.")
        return

    df = sentiment_df.sort_values("date")

    st.line_chart(
        df.set_index("date")["sentiment_score"]
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Average Sentiment",
        round(df["sentiment_score"].mean(), 3)
    )

    col2.metric(
        "Latest Sentiment",
        round(df["sentiment_score"].iloc[-1], 3)
    )


def show_recent_news():

    st.subheader("Recent Financial News")

    if news_df.empty:
        st.warning("No news data available.")
        return

    recent = news_df.sort_values(
        "created",
        ascending=False
    ).head(5)

    for _, row in recent.iterrows():

        st.markdown(f"**{row['title']}**")

        st.caption(row["created"])

        st.write(row["url"])

        st.divider()

# ----------------------------
# PAGE LAYOUT WITH TABS
# ----------------------------

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Forecasting",
    "Sentiment Analysis"
])

# ----------------------------
# OVERVIEW TAB
# ----------------------------

with tab1:

    st.header(company_name)

    show_metrics()

# ----------------------------
# FORECASTING TAB
# ----------------------------

with tab2:

    show_historical_plot()

    st.divider()

    show_yearly_plots()

    st.divider()

    show_future_plot()

# ----------------------------
# SENTIMENT TAB
# ----------------------------

with tab3:

    show_sentiment_trend()

    st.divider()

    show_recent_news()