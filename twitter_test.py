import time
import tweepy

# ✅ Replace with your actual Bearer Token
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAA5A0AEAAAAAQkKtD2DlRpCc%2F5CVdjhO8M0vAxA%3DfLZFiirpZssNp97Z3sb6hDJ1yT0FQOXvmUbgJmccu7Wx2OZe0N"

# ✅ Authenticate
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# ✅ Define Query
query = "AI"

try:
    # ✅ Reduce frequency of requests (add delay)
    time.sleep(2)  # Wait 5 seconds before making a request
    tweets = client.search_recent_tweets(query=query, max_results=10, tweet_fields=["author_id", "created_at"])

    # ✅ Print Tweets
    if tweets.data:
        for tweet in tweets.data:
            print(f"[{tweet.created_at}] User ID {tweet.author_id}: {tweet.text}\n")
    else:
        print("No tweets found.")

except tweepy.TooManyRequests:
    print("❌ Too many requests! Try again later.")

print("✅ Done!")
