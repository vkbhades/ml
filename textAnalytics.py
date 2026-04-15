from textblob import TextBlob

text = "The movie was alright"

analysis = TextBlob(text)

print("Text:", text)
print("Sentiment Polarity:", analysis.sentiment.polarity)

if analysis.sentiment.polarity > 0:
    print("Sentiment: Positive")
elif analysis.sentiment.polarity < 0:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")