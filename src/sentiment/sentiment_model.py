import pandas as pd
import os
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


INPUT_PATH = "../../data/processed_news/cleaned_news.csv"
OUTPUT_PATH = "../../data/sentiment/news_sentiment_scores.csv"


def generate_sentiment():

    print("Generating sentiment scores...")

    nltk.download("vader_lexicon")

    df = pd.read_csv(INPUT_PATH)

    sia = SentimentIntensityAnalyzer()

    df["sentiment_score"] = df["clean_text"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    os.makedirs("../../data/sentiment", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Sentiment scores saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_sentiment()