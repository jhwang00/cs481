import pandas as pd
from collections import Counter

def bag_of_words(data):
    # Get the unique words in the dataset
    vocab = set()
    for text in data['text']:
        words = text.split()
        vocab.update(words)
        
    # Calculate the word frequencies for each tweet
    word_freqs = []
    for i, text in enumerate(data['text']):
        words = text.split()
        freqs = Counter(words)
        # Add-one smoothing
        freqs.update({word: 1 for word in vocab if word not in freqs})
        row = {'id': i, 'freqs': freqs, 'label': data.loc[i, 'airline_sentiment']}
        word_freqs.append(row)
        
    # Convert the word frequencies to a bag of words vector for each tweet
    X = []
    y = []
    for row in word_freqs:
        vector = [row['freqs'][word] for word in vocab]
        X.append(vector)
        y.append(row['label'])
        
    return X, y, list(vocab)



# import pandas as pd
# from collections import Counter

# def bag_of_words(data):
#     # Get the unique words in the dataset
#     vocab = set()
#     for text in data['text']:
#         words = text.split()
#         vocab.update(words)
        
#     # Calculate the word frequencies for each tweet
#     word_freqs = []
#     for i, text in enumerate(data['text']):
#         words = text.split()
#         freqs = Counter(words)
#         row = {'id': i, 'freqs': freqs, 'label': data.loc[i, 'airline_sentiment']}
#         word_freqs.append(row)
        
#     # Convert the word frequencies to a bag of words vector for each tweet
#     X = []
#     y = []
#     for row in word_freqs:
#         vector = [row['freqs'][word] for word in vocab]
#         X.append(vector)
#         y.append(row['label'])
        
#     return X, y, list(vocab)

# from sklearn.feature_extraction.text import CountVectorizer

# def bag_of_words(data, vocab):
#     # Takes vocabulary list as additional input
#     # Extract the text and labels from dataframe
#     X_text = data['text']
#     y = data['airline_sentiment']
    
#     # Create a CountVectorizer object and fit it to the training data
#     if vocab is None:
#         vectorizer = CountVectorizer()
#     else:
#         vectorizer = CountVectorizer(vocab = vocab)
#     vectorizer.fit(X_text)
    
#     # Convert text data to a Bag of Words representation
#     X = vectorizer.transform(X_text)
    
#     return X, y, vectorizer.get_feature_names()

