import nltk
from nltk.corpus import stopwords

def preprocessLC(text):
    corp = text.lower()
    return corp

def preprocessStop(text):
    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    removal_list = list(stop_words) + ['lt','rt']
    # converts the words in text to lower case and then checks whether 
    #they are present in stop_words or not
    filtered_text = [w for w in text if not w.lower() in removal_list]
  
    for w in text:
        if w not in removal_list:
            filtered_text.append(w)

    return filtered_text




