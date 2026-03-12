import pandas as pd
import re
import os


INPUT_PATH = "../../data/raw_news/benzinga_news_raw.csv"
OUTPUT_PATH = "../../data/processed_news/cleaned_news.csv"


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove punctuation / numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_news():

    print("Cleaning news text...")

    df = pd.read_csv(INPUT_PATH)

    # combine title + body for better sentiment context
    df["full_text"] = df["title"].fillna("") + " " + df["body"].fillna("")

    df["clean_text"] = df["full_text"].apply(clean_text)

    os.makedirs("../data/processed_news", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved cleaned news to {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_news()