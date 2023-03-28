import csv
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#import string

t = []

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
            if text[i][j] not in vocab:
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

#def count(voca, sentence):
#    l = []
#    for j in range(len(voca)):
#        exists = 0
#        if voca[j] in sentence:
#            exists = 1
#        if exists == 0:
#            l.append(0)
#        else:
#            l.append(1)
#    return l

def positive(word, voca, pos): #input every positive (word, vocabulary, sentence list)
    bag_of_words = []
    for i in range(len(pos)):
        bag_of_words.append(bow(voca, pos[i]))
    pos_count = len(voca)
    for i in range(len(pos)):
        pos_count += len(pos[i])
    num = 1
    for i in range(len(bag_of_words)):
        if bag_of_words[i][word] == 1:
            num+=1
    return num/pos_count

def negative(word, voca, neg):
    neg_count = 0
    for i in range(len(neg)):
        neg_count += len(neg[i])
    

def neutral(voca, neu):
    neu_count = 0    
    for i in range(len(neu)):
        neu_count += len(neu[i])

def main():
    n = 11 #int((len(t)-1)*80/100)
    l = [] #test set list
    pos = []
    neu = []
    neg = []
    for i in range(1, n):
        l.append(preprocessStop(preprocessStem(preprocessLC(t[i][10]))))
    vocabulary = vocabb(l)
    for i in range(1, n):
        if t[i][1] == 'positive':
            pos.append(preprocessStop(preprocessStem(preprocessLC(t[i][10]))))
        elif t[i][1] == 'negative':
            neg.append(preprocessStop(preprocessStem(preprocessLC(t[i][10]))))
        else:
            neu.append(preprocessStop(preprocessStem(preprocessLC(t[i][10]))))
    
    print(positive('@virginamerica', vocabulary, pos))

    #P(pos) = len(pos)/len(l)
    #P(neg) = len(neg)/len(l)
    #P(neu) = len(neu)/len(l)

#positive
#negative
#neutral
# t[0] :: column name
# t[i][1] ::sentiment
# t[i][10] ::text
    

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
