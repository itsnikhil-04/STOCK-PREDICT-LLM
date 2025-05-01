import tweepy
import time
import pandas as pd
from transformers import pipeline

# Twitter API credentials
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAA5A0AEAAAAAQkKtD2DlRpCc%2F5CVdjhO8M0vAxA%3DfLZFiirpZssNp97Z3sb6hDJ1yT0FQOXvmUbgJmccu7Wx2OZe0N"

# Authenticate with Twitter API
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Define the stock-related search query
query = "(AAPL OR TSLA OR AMZN OR GOOGL OR stock market) -is:retweet lang:en"

# Function to fetch tweets
def fetch_tweets(query, max_results=10):
    try:
        tweets = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["author_id", "created_at"])
        return tweets
    except tweepy.TooManyRequests:
        print("Too many requests! Waiting before retrying...")
        time.sleep(900)  # Wait 15 minutes before retrying
        return fetch_tweets(query, max_results)
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to summarize tweets using an offline model (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_tweets_local(texts):
    """Summarizes stock tweets using the BART model."""
    input_text = " ".join(texts[:10])  # Take first 10 tweets
    return summarizer(input_text, max_length=100, min_length=20, do_sample=False)[0]["summary_text"]

# Fetch tweets
tweets = fetch_tweets(query, max_results=10)

# Save tweets to CSV
data = []
if tweets and tweets.data:
    for tweet in tweets.data:
        data.append([tweet.id, tweet.created_at, tweet.author_id, tweet.text])
    
    df = pd.DataFrame(data, columns=["Tweet_ID", "Created_At", "Author_ID", "Text"])
    df.to_csv("stock_tweets.csv", index=False)
    print("✅ Tweets saved to stock_tweets.csv")
    
    # Summarize tweets using BART
    tweet_texts = df["Text"].tolist()
    summary = summarize_tweets_local(tweet_texts)
    
    # Save summary to CSV
    summary_df = pd.DataFrame([[summary]], columns=["Summary"])
    summary_df.to_csv("tweet_summary.csv", index=False)
    print("Summary saved to tweet_summary.csv")
    
    print(" Summary:\n", summary)
else:
    print("No tweets found.")
