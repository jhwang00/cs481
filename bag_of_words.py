from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(data):
    # Extract the text and labels from dataframe
    X_text = data['text']
    y = data['airline_sentiment']
    
    # Create a CountVectorizer object and fit it to the training data
    vectorizer = CountVectorizer()
    vectorizer.fit(X_text)
    
    # Convert text data to a Bag of Words representation
    X = vectorizer.transform(X_text)
    
    return X, y, vectorizer.get_feature_names()
