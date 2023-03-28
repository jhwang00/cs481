import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Tweets.csv')

# Separate features and labels
X = df['text']
y = df['airline_sentiment']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bag of Words function
def bag_of_words(vocab, sentence):
    vectorizer = CountVectorizer(vocabulary=vocab)
    bow = vectorizer.fit_transform([sentence])
    return bow.toarray()

# Build Bag of Words vocabulary
vocab = CountVectorizer().fit(X_train).vocabulary_

# Extract Bag of Words features for training and testing sets
X_train_bow = [bag_of_words(vocab, sentence) for sentence in X_train]
X_test_bow = [bag_of_words(vocab, sentence) for sentence in X_test]

# Train Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train_bow, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test_bow)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification report:\n{report}')

