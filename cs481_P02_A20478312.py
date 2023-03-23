import csv
import sys
import nltk
from nltk.corpus import stopwords
#import string

t = []
vocab = []
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

# int((len(t[i])-1)*80/100)

def preprocessLC(text): #lowercase
    corp = text.lower().split()
    return corp

def preprocessStem(text): #stemming
    stemmer = nltk.stem.LancasterStemmer()
    text = text.split()
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

def vocabb():
    s = input("Enter sentence or type 'end': ")
    if s == 'end':
        print(vocab)
        tot = 0
        for k in range(len(vocab)):
            tot += vocab[k][1]
        print(tot)
        return
    else:
        s = s.split()
        for i in range(len(s)):
            if vocab == []:
                vocab.append([s[i],1])
            else:
                exists = 0
                for j in range(len(vocab)):
                    if s[i] == vocab[j][0]:
                        exists = 1
                        vocab[j][1] += 1
                if exists != 1:
                    vocab.append([s[i], 1])
        #print(vocab)
        vocabb()

def main():
    l = []
    for i in range(1, 3):
        l.append(preprocessStop((preprocessLC(t[i][10]))))

    print(l)
    #print(preprocessStop(l))
    #try:
    #    arg = sys.argv[1]
    #    if arg == 'NO':
    #        pass # do stemming
    #except:
    #    pass # skip stemming
    #    print("\nERROR: Not enough or too many input arguments.\n")
    

main()
