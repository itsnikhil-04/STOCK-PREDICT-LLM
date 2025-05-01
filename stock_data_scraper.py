import tweepy
import time
import pandas as pd
import yfinance as yf
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

# List of stock symbols to process
stocks = ["AAPL", "TSLA", "AMZN", "GOOGL"]
training_data_list = []

for stock_symbol in stocks:
    print(f" Processing {stock_symbol}...")
    
    # Fetch tweets
    tweets = fetch_tweets(f"({stock_symbol} stock) -is:retweet lang:en", max_results=10)
    
    if tweets and tweets.data:
        data = [[tweet.id, tweet.created_at, tweet.author_id, tweet.text] for tweet in tweets.data]
        df = pd.DataFrame(data, columns=["Tweet_ID", "Created_At", "Author_ID", "Text"])
        
        # Summarize tweets using BART
        tweet_texts = df["Text"].tolist()
        summary = summarize_tweets_local(tweet_texts)
        
        # Fetch stock price movement
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="10d")  # Get last 10 days of stock data
        
        if len(hist) >= 2:
            prev_close = hist["Close"].iloc[-2]
            curr_close = hist["Close"].iloc[-1]
            if curr_close > prev_close:
                price_movement = "Up"
            elif curr_close < prev_close:
                price_movement = "Down"
            else:
                price_movement = "Neutral"
            
            # Save training data
            training_data = {"prompt": f"Stock summary: {summary} Predict the movement.", "completion": price_movement}
            training_data_list.append(training_data)
        else:
            print(f" Not enough stock data for {stock_symbol}.")
    else:
        print(f" No tweets found for {stock_symbol}.")

# Save all training data to JSONL file
if training_data_list:
    train_df = pd.DataFrame(training_data_list)
    train_df.to_json("stock_training_data.jsonl", orient="records", lines=True)
    print(" Training data saved as stock_training_data.jsonl")
else:
    print(" No valid training data collected.")
