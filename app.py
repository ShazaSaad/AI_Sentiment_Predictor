import streamlit as st
import pandas as pd
import os
from PIL import Image

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="AI Stock Intelligence",
    layout="wide"
)

# ----------------------------
# LIGHT FINTECH UI (CUSTOM CSS)
# ----------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    color: #111827;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Cards */
[data-testid="metric-container"],
.custom-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
}

/* Headers */
h1, h2, h3 {
    color: #111827;
}

/* Divider */
hr {
    border: 1px solid #e5e7eb;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    background-color: #4f46e5;
    color: white;
    border: none;
}

/* News Cards */
.news-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin-bottom: 12px;
    transition: 0.2s ease;
}

.news-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

/* Tag badge */
.badge {
    display: inline-block;
    padding: 4px 10px;
    background: #eef2ff;
    color: #4338ca;
    border-radius: 999px;
    font-size: 12px;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER (PRODUCT STYLE)
# ----------------------------

st.markdown("""
# 📊 AI Stock Intelligence
### Smarter Forecasting • Sentiment-Aware Insights • Decision Support
""")

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
FUTURE_PLOTS = "forecasting_pipeline/plots/future"
SENTIMENT_PATH = "src/data/sentiment/daily_sentiment_scores.csv"
NEWS_PATH = "src/data/raw_news/benzinga_news_raw.csv"

# ----------------------------
# DATA LOADING
# ----------------------------

@st.cache_data
def load_metrics():
    return pd.read_csv(RESULTS_PATH)

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

with st.spinner("Loading intelligence modules..."):
    results_df = load_metrics()
    sentiment_df = load_sentiment()
    news_df = load_news()

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.markdown("## 🔎 Controls")

company_name = st.sidebar.selectbox(
    "Select Company",
    list(companies.keys())
)

ticker = companies[company_name]

st.sidebar.markdown(f"""
<div class="custom-card">
<b>Selected:</b><br>
{company_name}<br><br>
<b>Ticker:</b> {ticker}
</div>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER SECTION
# ----------------------------

st.markdown(f"## {company_name} ({ticker})")

# ----------------------------
# METRICS
# ----------------------------

def show_metrics():
    df = results_df[results_df["ticker"] == ticker]

    if df.empty:
        st.warning("No evaluation results found.")
        return

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", f"{df['R2'].mean():.3f}")
    col2.metric("RMSE", f"{df['RMSE'].mean():.3f}")
    col3.metric("Directional Accuracy", f"{df['Directional_Accuracy'].mean():.2%}")

# ----------------------------
# FORECASTING
# ----------------------------

def show_future_plot():
    file_path = f"{FUTURE_PLOTS}/{ticker}_forecast.png"

    if os.path.exists(file_path):
        st.image(file_path, width='stretch', caption="3-Month Forecast")
    else:
        st.info("Forecast not available.")

def show_yearly_plots():
    if not os.path.exists(YEARLY_PLOTS):
        st.warning("Yearly plots folder missing.")
        return

    files = sorted(os.listdir(YEARLY_PLOTS))

    for file in files:
        if file.startswith(ticker):
            img = Image.open(os.path.join(YEARLY_PLOTS, file))
            st.image(img, caption=file, width='stretch')

# ----------------------------
# SENTIMENT
# ----------------------------

def show_sentiment():
    if sentiment_df.empty:
        st.warning("Sentiment data missing.")
        return

    df = sentiment_df.sort_values("date")

    st.line_chart(
        df.set_index("date")["sentiment_score"],
        height=300
    )

    col1, col2 = st.columns(2)
    col1.metric("Average Sentiment", f"{df['sentiment_score'].mean():.3f}")
    col2.metric("Latest Sentiment", f"{df['sentiment_score'].iloc[-1]:.3f}")

# ----------------------------
# NEWS
# ----------------------------

def show_news():
    if news_df.empty:
        st.warning("No news data available.")
        return

    recent = news_df.sort_values("created", ascending=False).head(5)

    for _, row in recent.iterrows():
        st.markdown(f"""
<div class="news-card">
<div class="badge">Market News</div>
<h4>{row['title']}</h4>
<p style="font-size:13px;color:gray;">{row['created']}</p>
<a href="{row['url']}" target="_blank">Read full article →</a>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# TABS
# ----------------------------

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Forecasting",
    "Sentiment"
])

# ----------------------------
# OVERVIEW
# ----------------------------

with tab1:

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 Model Performance")
        show_metrics()

    with col2:
        st.markdown("### 🧠 Insights")
        st.markdown("""
<div class="custom-card">
• AI-driven predictions  
• Uses sentiment + historical data  
• Tracks directional accuracy  
• Built for decision support  
</div>
""", unsafe_allow_html=True)

# ----------------------------
# FORECASTING
# ----------------------------

with tab2:

    st.markdown("### 📊 Future Outlook")

    col1, col2 = st.columns([2, 1])

    with col1:
        show_future_plot()

    with col2:
        st.markdown("""
<div class="custom-card">
<b>Forecast Window:</b> 3 Months<br><br>
<b>Signals Used:</b>
<ul>
<li>Historical price trends</li>
<li>Market sentiment</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📅 Historical Forecasts")
    show_yearly_plots()

# ----------------------------
# SENTIMENT
# ----------------------------

with tab3:

    st.markdown("### 🧠 Sentiment Trends")
    show_sentiment()

    st.divider()

    st.markdown("### 📰 Latest News")
    show_news()