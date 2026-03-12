from data_collection.fetch_benzinga_news import fetch_benzinga_news
from preprocessing.clean_news import clean_news
from sentiment.sentiment_model import generate_sentiment
from sentiment.sentiment_pipeline import create_daily_sentiment

def run_pipeline():

    print("Step 1: Fetching news...")
    fetch_benzinga_news()

    print("Step 2: Cleaning text...")
    clean_news()

    print("Step 3: Running sentiment model...")
    generate_sentiment()

    print("Step 4: Creating daily sentiment scores...")
    create_daily_sentiment()

    print("Pipeline complete!")

if __name__ == "__main__":

    run_pipeline()