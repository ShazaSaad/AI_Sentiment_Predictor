import streamlit as st
import pandas as pd
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Stock Forecasting Dashboard",
    layout="wide"
)

st.title("Stock Forecasting & Sentiment Analysis Dashboard")

# Companies list
companies = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "Alphabet (GOOG)": "GOOG",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "Eli Lilly (LLY)": "LLY",
    "Broadcom (AVGO)": "AVGO",
    "ASML (ASML)": "ASML",
    "Taiwan Semiconductor (TSM)": "TSM",
    "JPMorgan Chase (JPM)": "JPM",
    "Visa (V)": "V",
    "Mastercard (MA)": "MA",
    "Walmart (WMT)": "WMT",
    "ExxonMobil (XOM)": "XOM",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Novo Nordisk (NVO)": "NVO",
    "Procter & Gamble (PG)": "PG",
    "Costco (COST)": "COST",
    "Home Depot (HD)": "HD",
    "Coca-Cola (KO)": "KO",
    "PepsiCo (PEP)": "PEP",
    "Toyota (TM)": "TM"
}

# Path configuration
RESULTS_PATH = "forecasting_pipeline/evaluation_results.csv"

YEARLY_PLOTS = "forecasting_pipeline/plots/yearly"
HISTORICAL_PLOTS = "forecasting_pipeline/plots/historical"
FUTURE_PLOTS = "forecasting_pipeline/plots/future"

# Loading metrics
@st.cache_data
def load_metrics():
    return pd.read_csv(RESULTS_PATH)

results_df = load_metrics()

# Building a sidebar to select companies
st.sidebar.header("Select Company")
company_name = st.sidebar.selectbox("Company", list(companies.keys()))
ticker = companies[company_name]
st.sidebar.write("Ticker:", ticker)

# Building a function to show the metrics results

def show_metrics(ticker):
    st.header("Model Evaluation Metrics")
    company_results = results_df[
        results_df["ticker"] == ticker
    ]
    if company_results.empty:
        st.warning("No evaluation results found.")
        return
    st.dataframe(company_results)
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


# Function to show yearly forecasts
def show_yearly_plots(ticker):
    st.header("Yearly Forecasts (Jan–Mar Predictions)")
    
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
        st.info("No yearly plots available.")

# Function to show historical forecasts

def show_historical_plot(ticker):

    st.header("Past 6 Years + Prediction")

    file_path = f"{HISTORICAL_PLOTS}/{ticker}_history.png"

    if os.path.exists(file_path):

        st.image(file_path)

    else:
        st.info("Historical plot not available.")

# Function to show future forecasts
def show_future_plot(ticker):
    st.header("Next 3 Month Forecast")
    file_path = f"{FUTURE_PLOTS}/{ticker}_future.png"
    
    if os.path.exists(file_path):
        st.image(file_path)
    else:
        st.info("Future forecast not available.")

# Main display
st.subheader(company_name)
show_metrics(ticker)
show_yearly_plots(ticker)
show_historical_plot(ticker)
show_future_plot(ticker)

