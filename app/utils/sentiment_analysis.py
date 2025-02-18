from newsapi import NewsApiClient
from textblob import TextBlob
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import os
import dotenv

dotenv.load_dotenv()

api_key = os.getenv('NEWS_API_KEY')

class NewsAnalyzer:
    def __init__(self, api_key=api_key):  # Replace with your NewsAPI key
        self.newsapi = NewsApiClient(api_key=api_key)
        
    def fetch_news(self, symbol, company_name, days=7):
        # Get news for both symbol and company name for better coverage
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        articles = []
        
        # Search using both stock symbol and company name
        for query in [symbol, company_name]:
            try:
                news = self.newsapi.get_everything(
                    q=query,
                    from_param=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='publishedAt'
                )
                
                if news['status'] == 'ok':
                    articles.extend(news['articles'])
            except Exception as e:
                st.warning(f"Error fetching news for {query}: {str(e)}")
        
        return articles

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def get_sentiment_summary(self, articles):
        if not articles:
            return {
                'overall_sentiment': 0,
                'positive_news': 0,
                'negative_news': 0,
                'neutral_news': 0,
                'sentiment_data': pd.DataFrame()
            }

        sentiments = []
        dates = []
        headlines = []
        
        for article in articles:
            sentiment = self.analyze_sentiment(article['title'] + ' ' + (article['description'] or ''))
            sentiments.append(sentiment)
            dates.append(pd.to_datetime(article['publishedAt']))
            headlines.append(article['title'])

        sentiment_df = pd.DataFrame({
            'date': dates,
            'sentiment': sentiments,
            'headline': headlines
        })
        
        return {
            'overall_sentiment': np.mean(sentiments),
            'positive_news': len([s for s in sentiments if s > 0.1]),
            'negative_news': len([s for s in sentiments if s < -0.1]),
            'neutral_news': len([s for s in sentiments if -0.1 <= s <= 0.1]),
            'sentiment_data': sentiment_df
        }
