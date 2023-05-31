import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess(tweet):
    # Convert text to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    
    # Remove punctuation and special characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    # Tokenize the text into individual words
    words = word_tokenize(tweet)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return words

# Load the dataset
data = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None)

# Assign column names to the dataset
data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Print the first few rows of the dataset
print(data.head())

# Create an instance of the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Initialize an empty dictionary
sentiments = {}
word_counts = {}

# text = "I love this product! It's amazing."

# Perform sentiment analysis
# sentiment = sia.polarity_scores(text)

for tweet in data['text']:
    sentiment = sia.polarity_scores(tweet)
    
    # Determine the sentiment category based on the compound score
    if sentiment['compound'] >= 0.05:
        sentiment_category = 'positive'
    elif sentiment['compound'] <= -0.05:
        sentiment_category = 'negative'
    else:
        sentiment_category = 'neutral'
    
    # Update the dictionary with the sentiment category
    if sentiment_category in sentiments:
        sentiments[sentiment_category] += 1
    else:
        sentiments[sentiment_category] = 1

    words = preprocess(tweet)

    # Update word counts and sentiment
    for word in words:
        if word in word_counts:
            word_counts[word][0] += sentiment['pos']
            word_counts[word][1] += sentiment['neg']
        else:
            word_counts[word] = [sentiment['pos'], sentiment['neg']]

    

print(sentiments)

# Pie chart labels and sizes
labels = sentiments.keys()
sizes = sentiments.values()

sorted_word_counts = sorted(word_counts.items(), key=lambda x: sum(x[1]), reverse=True)

# Select the top N words to plot
top_n = 10
top_words = [word[0] for word in sorted_word_counts[:top_n]]
positive_counts = [word_counts[word][0] for word in top_words]
negative_counts = [word_counts[word][1] for word in top_words]

# Create the pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%')

# Add a title
plt.title('Sentiment Analysis Results')

# Create the bar graph
fig, ax = plt.subplots()
index = range(top_n)
bar_width = 0.35
opacity = 0.8

rects1 = ax.bar(index, positive_counts, bar_width,
                alpha=opacity, color='g', label='Positive')

rects2 = ax.bar(index, negative_counts, bar_width,
                alpha=opacity, color='r', label='Negative', bottom=positive_counts)

ax.set_xlabel('Words')
ax.set_ylabel('Count')
ax.set_title('Most Used Words and Sentiment')
ax.set_xticks(index)
ax.set_xticklabels(top_words, rotation=45)
ax.legend()

plt.tight_layout()

# Display the chart
plt.show()

# Perform sentiment analysis on each tweet
# for tweet in data['text']:
#     sentiment = sia.polarity_scores(tweet)
#     print(tweet)
#     print(sentiment)