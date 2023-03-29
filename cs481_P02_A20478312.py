import csv
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import math

t = []

with open ('Tweets.csv', 'r', encoding='utf-8') as csv_file:
    csv_data = csv.reader(csv_file)
    print(csv_data)
    for data in csv_data:
        t.append(data)


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

def positive(word, voca, pos): #input every positive (word, vocabulary, sentence list) returns p(word|pos)
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
    return math.log(num/pos_count)

def negative(word, voca, neg):
    bag_of_words = []
    for i in range(len(neg)):
        bag_of_words.append(bow(voca, neg[i]))
    neg_count = 0
    for i in range(len(neg)):
        neg_count += len(neg[i])
    num = 1
    for i in range(len(bag_of_words)):
        if bag_of_words[i][word] == 1:
            num+=1
    return math.log(num/neg_count)

def neutral(word, voca, neu):
    bag_of_words = []
    for i in range(len(neu)):
        bag_of_words.append(bow(voca, neu[i]))
    neu_count = 0    
    for i in range(len(neu)):
        neu_count += len(neu[i])
    num = 1
    for i in range(len(bag_of_words)):
        if bag_of_words[i][word] == 1:
            num+=1
    return math.log(num/neu_count)

def matrix(mat_1, mat_2, mat_3, real_value, classified_value):
    if classified_value =='positive':
        if real_value == 'positive':
            mat_1[1] += 1
        elif real_value == 'negative':
            mat_1[2] += 1
        else:
            mat_1[3] += 1
    elif classified_value == 'negative':
        if real_value == 'positive':
            mat_2[1] += 1
        elif real_value == 'negative':
            mat_2[2] += 1
        else:
            mat_2[3] += 1
    else:
        if real_value == 'positive':
            mat_3[1] += 1
        elif real_value == 'negative':
            mat_3[2] += 1
        else:
            mat_3[3] += 1

def main():
    n = int((len(t)-1)*80/100)
    boundary = int((len(t)-1)*80/100)
    train_set = [] #train set list
    test_set = [] #test set list
    pos = []
    neu = []
    neg = []

    mat_0 = ["system\\real", "positive", "negative", "neutral"]
    mat_1 = ["positive", 0, 0, 0]
    mat_2 = ["negative", 0, 0, 0]
    mat_3 = ["neutral", 0, 0, 0]

    for i in range(1, n):
        cleaned_train = preprocessStop(preprocessStem(preprocessLC(t[i][10])))
        train_set.append(cleaned_train)
    vocabulary = vocabb(train_set)
    for i in range(1, n):
        if t[i][1] == 'positive':
            pos.append(cleaned_train)
        elif t[i][1] == 'negative':
            neg.append(cleaned_train)
        else:
            neu.append(cleaned_train)
    prob_pos = len(pos)/len(train_set)
    prob_neg = len(neg)/len(train_set)
    prob_neu = len(neu)/len(train_set)

    prob_pos_word = {}
    prob_neg_word = {}
    prob_neu_word = {}

    for i in range(len(vocabulary)): # store probabilities
        prob_pos_word[vocabulary[i]] = (positive(vocabulary[i], vocabulary, pos))
        prob_neg_word[vocabulary[i]] = (positive(vocabulary[i], vocabulary, neg))
        prob_neu_word[vocabulary[i]] = (positive(vocabulary[i], vocabulary, neu))

    for i in range(boundary+1, len(t)):
        cleaned_test = preprocessStop(preprocessStem(preprocessLC(t[i][10])))
        test_set.append(cleaned_test)
    print("training set done")

    for i in range(boundary+1, boundary+21): # test set tesing  len(t)
        is_pos = math.log(prob_pos)
        is_neg = math.log(prob_neg)
        is_neu = math.log(prob_neu)
        index = i-1-boundary
        for j in range(len(test_set[index])): #positive probability
            is_pos = is_pos + prob_pos_word[test_set[index][j]]
            print(is_pos)
        print("final", is_pos)
        for j in range(len(test_set[index])): #negative probability
            is_neg = is_neg + prob_neg_word[test_set[index][j]]
            print(is_pos)
        print("final", is_neg)
        for j in range(len(test_set[index])): #neutral probability
            is_neu = is_neu + prob_neu_word[test_set[index][j]]
            print(is_pos)
        print("final", is_neu)

        classification = max(is_pos, is_neg, is_neu) # classification based on trained set

        matrix(mat_1, mat_2, mat_3, t[i][1], classification) # put it in confusion matrix
        print("done")
        print(mat_0)
        print(mat_1)
        print(mat_2)
        print(mat_3)
    
    print(mat_0)
    print(mat_1)
    print(mat_2)
    print(mat_3)
        #compare with real value
        #if TP, TN, FP, FN
        #calculate recall, specificity, etc

        #keyboard input(single sentence)
    

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
