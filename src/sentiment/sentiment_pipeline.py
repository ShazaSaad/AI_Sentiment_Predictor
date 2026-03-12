import pandas as pd
import os


INPUT_PATH = "../../data/sentiment/news_sentiment_scores.csv"
OUTPUT_PATH = "../../data/sentiment/daily_sentiment_scores.csv"


def create_daily_sentiment():

    print("Creating daily sentiment scores...")

    df = pd.read_csv(INPUT_PATH)

    df["created"] = pd.to_datetime(df["created"], errors="coerce")

    df["date"] = df["created"].dt.date

    daily_sentiment = (
        df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
    )

    os.makedirs("../../data/sentiment", exist_ok=True)

    daily_sentiment.to_csv(OUTPUT_PATH, index=False)

    print(f"Daily sentiment saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    create_daily_sentiment()