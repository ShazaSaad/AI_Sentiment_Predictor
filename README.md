ai-stock-sentiment-project/
│
├── data/
│   ├── raw_news/
│   │   └── benzinga_news_raw.csv
│   │
│   ├── processed_news/
│   │   └── cleaned_news.csv
│   │
│   └── sentiment/
│       └── daily_sentiment_scores.csv
│
├── src/
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
│   │
│   └── main_sentiment_pipeline.py
│
└── README.md
