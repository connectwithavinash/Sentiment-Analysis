from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import unicodedata
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load data from csv
df = pd.read_csv('/Users/Abhi/Desktop/Innomatics/Intership /Sentiment_Analysis/Data/reviews_badminton/data.csv')

# Removing duplicate & null values
df.drop_duplicates(inplace=True) 
df.dropna(subset=["Review text"],inplace=True)

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]+', ' ', text)

    # Convert text to lowercase
    text = text.lower()

    # Remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Expand contractions
    text = contractions.fix(text)

    # Tokenize text into words
    words = word_tokenize(text)
    
    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    
    wnl = WordNetLemmatizer()
    # Lemmatize words
    words = [wnl.lemmatize(word) for word in words]

    # Join words back into a single string
    text = ' '.join(words)

    return pd.Series([text])

df["Tokens"] = df["Review text"].apply(lambda x:preprocess_text(x))

######
Label = []
for i in df["Ratings"]:
    if i >= 4:
        Label.append(1)
    else:
        Label.append(0)

df["Label"] = Label

#Train Test Split 
from sklearn.model_selection import train_test_split
train,test = train_test_split(df, test_size=0.20, random_state=42)

train_tokens = []
for i in train["Tokens"]:
    train_tokens.append(i)

test_tokens = []
for i in test["Tokens"]:
    test_tokens.append(i)


# Feature Extraction BoW model
bow_vectorizer = CountVectorizer()
train_features = bow_vectorizer.fit_transform(train_tokens)
test_features = bow_vectorizer.transform(test_tokens)


#Model Training
bow_clf = MultinomialNB()
bow_clf.fit(train_features, train["Label"])


# Define function to predict sentiment of a review using BoW model
def predict_sentiment_bow(review):
    # Preprocess review text
    review = preprocess_text(review)

    review = review.to_string()
    # Encode review text as BoW features
    review_features = bow_vectorizer.transform([review])

    # Make prediction using BoW model
    predicted_class = bow_clf.predict(review_features)

    # Return predicted sentiment as a string
    if predicted_class == 0:
        return 'Negative'
    else:
        return 'Positive'


@app.route('/')
def index():
    return render_template('index.html')
# Define route to accept user reviews and predict their sentiment using BoW model
@app.route('/sentiment', methods=['POST'])
def sentiment():
    review = request.form.get('review')
    sentiment = predict_sentiment_bow(review)
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(host=0.0.0.0,debug=True,port=5000)
