import csv
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#import string

t = []
pos = []
neut = []
neg = []

with open ('Tweets.csv', 'r', encoding='utf-8') as csv_file:
    csv_data = csv.reader(csv_file)
    print(csv_data)
    for data in csv_data:
        t.append(data)
#positive
#negative
#neutral
# t[0] :: column name
# t[i][1] ::sentiment
# t[i][10] ::text

# int((len(t)-1)*80/100)

def preprocessLC(text): #lowercase
    corp = text.lower().split()
    return corp

def preprocessStem(text): #stemming
    stemmer = nltk.stem.LancasterStemmer()
    text = text
    stemmedWord = []
    for w in text:
        stemmedWord.append(stemmer.stem(w))
    return stemmedWord

def preprocessStop(text): #stopwords
    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    removal_list = list(stop_words) + ['lt','rt']
    # converts the words in text to lower case and then checks whether 
    #they are present in stop_words or not
    filtered_text = [w for w in text if not w.lower() in removal_list]
    return filtered_text

def vocabb(text): #extracting vocab list from training set(text)
    vocab = []
    for i in range(len(text)):
        for j in range(len(text[i])):
            if not vocab:
                exists = 0
                for j in range(len(vocab)):
                    if text[i][j] in vocab:
                        exists = 1
                if exists == 0:
                    vocab.append(text[i][j])
            else:
                vocab.append(text[i][j])
    return vocab

#need number of pos/neut/neg
#need to count of occurence of each words in pos/neut/neg
#count total words (1s in bag of words)

def bow(voca, sentence):
    dict = {}
    for j in range(len(voca)):
        exists = 0
        if voca[j] in sentence:
            exists = 1
        if exists == 0:
            dict[voca[j]] = 0
        else:
            dict[voca[j]] = 1 
    return dict

def positive(text): #input every positive
    pass

def neutral():
    pass

def negative():
    pass

def main():
    voca = [] #vocabulary list
    l = [] #test set list
    for i in range(1, 6):
        l.append(preprocessStop(preprocessStem(preprocessLC(t[i][10]))))
        voca.append(preprocessStop(preprocessStem(preprocessLC(t[i][10]))))
    #print(l)
    #bag_of_words(t[i][10])
    vocabulary = vocabb(voca)
    #print(vocabulary)
    for i in range(len(l)):
        print(bow(vocabulary, l[i]))
    #print(int((len(t)-1)*80/100))

    #preprocess
    #train based on t[i][1]::sentiment
    #train
    #

    #print(l)
    #print(preprocessStop(l))
    #try:
    #    arg = sys.argv[1]
    #    if arg == 'NO':
    #        pass # do stemming
    #except:
    #    pass # skip stemming
    #    print("\nERROR: Not enough or too many input arguments.\n")
    

main()
