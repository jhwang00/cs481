import csv
from re import M
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import math
import time

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

def cleanText(text): #clean text before partitioning based on command line input
    for i in range(1, len(t)-1):
        if len(sys.argv) > 1:
            n = sys.argv[1]
            if n == "YES":
                clean_text = preprocessStop(preprocessLC(text))
                return clean_text
            else:
                clean_text = preprocessStop(preprocessStem(preprocessLC(text)))
                return clean_text
        else:
        clean_text = preprocessStop(preprocessStem(preprocessLC(text)))
        return clean_text

def vocabb(text): #extracting vocab list from training set(text)
    vocab = []
    for i in range(len(text)):
        for j in range(len(text[i])):
            if text[i][j] not in vocab:
                vocab.append(text[i][j])
    return vocab

def bow(voca, sentence):
    dict = {}
    count = 0
    for j in range(len(voca)):
        exists = 0
        if voca[j] in sentence:
            exists = 1
        if exists == 0:
            dict[voca[j]] = 0
        else:
            dict[voca[j]] = 1 
            count += 1
    return [dict, count]

def positive(word, voca, bag_of_words, pos_count): #input every positive (word, vocabulary, sentence list) returns p(word|pos)
    smoothing = len(voca) + pos_count
    num = 1
    for i in range(len(bag_of_words)):
        if bag_of_words[i][word] == 1:
            num+=1
    return math.log(num/smoothing)

def negative(word, voca, bag_of_words, neg_count):
    smoothing = len(voca) + neg_count
    num = 1
    for i in range(len(bag_of_words)):
        if bag_of_words[i][word] == 1:
            num+=1
    return math.log(num/smoothing)

def neutral(word, voca, bag_of_words, neu_count):
    smoothing = len(voca) + neu_count
    num = 1
    for i in range(len(bag_of_words)):
        if bag_of_words[i][word] == 1:
            num+=1
    return math.log(num/smoothing)

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
        cleaned_train = cleanText(t[i][10])
        train_set.append(cleaned_train)
    vocabulary = vocabb(train_set)
#voca preprocess done
    print("voca done\n")

    for i in range(len(train_set)):
        if t[i][1] == 'positive': #2003
            pos.append(train_set[i])
        elif t[i][1] == 'negative': #7090
            neg.append(train_set[i])
        else: #2618
            neu.append(train_set[i])
#classify done
    print("train set: ", len(train_set))
    print("positive: ", len(pos))
    print("negative: ", len(neg))
    print("neutral: ", len(neu))
    print("classification done\n")

    prob_pos = len(pos)/len(train_set) #probability of pos/neg/neu
    prob_neg = len(neg)/len(train_set)
    prob_neu = len(neu)/len(train_set)

    prob_pos_word = {} #probability of each word given pos/neg/neu
    prob_neg_word = {}
    prob_neu_word = {}

    pos_bag_of_words = [] # bag of words for pos/neg/neu
    neg_bag_of_words = []
    neu_bag_of_words = []

    pos_count = 0 # count(x, y) denominator
    neg_count = 0
    neu_count = 0

    for i in range(len(pos)):
        p = bow(vocabulary, pos[i])
        pos_bag_of_words.append(p[0])
        pos_count += p[1]
    for i in range(len(pos)):
        n = bow(vocabulary, neg[i])
        neg_bag_of_words.append(n[0])
        neg_count += n[1]
    for i in range(len(pos)):
        nn = p = bow(vocabulary, neu[i])
        neu_bag_of_words.append(nn[0])
        neu_count += nn[1]
#bow done
    print("bag of words done\n")

    for i in range(len(vocabulary)): # store probabilities
        prob_pos_word[vocabulary[i]] = (positive(vocabulary[i], vocabulary, pos_bag_of_words, pos_count))
        prob_neg_word[vocabulary[i]] = (negative(vocabulary[i], vocabulary, neg_bag_of_words, neg_count))
        prob_neu_word[vocabulary[i]] = (neutral(vocabulary[i], vocabulary, neu_bag_of_words, neu_count))
#probabilty of word done
    print("probability done\n")

    for i in range(boundary+1, len(t)):
        cleaned_test = cleanText(t[i][10])
        test_set.append(cleaned_test)
#test set
    print("test set preprocess done")
    print("test set: ", len(test_set))
    print()

######################################################
    for i in range(boundary+1, len(t)): # test set tesing  len(t)
        is_pos = math.log(prob_pos)
        is_neg = math.log(prob_neg)
        is_neu = math.log(prob_neu)
        index = i-1-boundary
        for j in range(len(test_set[index])): #positive probability
            if test_set[index][j] in vocabulary:
                is_pos = is_pos + prob_pos_word[test_set[index][j]]
        for j in range(len(test_set[index])): #negative probability
            if test_set[index][j] in vocabulary:
                is_neg = is_neg + prob_neg_word[test_set[index][j]]
        for j in range(len(test_set[index])): #neutral probability
            if test_set[index][j] in vocabulary:
                is_neu = is_neu + prob_neu_word[test_set[index][j]]
        classification = max(is_pos, is_neg, is_neu) # classification based on trained set
#checking classified value 
        if classification == is_pos:
            matrix(mat_1, mat_2, mat_3, t[i][1], "positive") # put it in confusion matrix
        elif classification == is_neg:
            matrix(mat_1, mat_2, mat_3, t[i][1], "negative")
        else:
            matrix(mat_1, mat_2, mat_3, t[i][1], "neutral")
        
#test done
    print("done\n")
    print(mat_0)
    print(mat_1)
    print(mat_2)
    print(mat_3)
    print("\n")
        #compare with real value
        #if TP, TN, FP, FN
        #calculate recall, specificity, etc

        #keyboard input(single sentence)
    

#06~
#20542 voca
# t[0] :: column name
# t[i][1] ::sentiment
# t[i][10] ::text
    

    #print(l)
    #print(preprocessStop(l))
    #try:
    #    arg = sys.argv[1]
    #    if arg == 'NO':
    #        pass # do stemming
    #except:
    #    pass # skip stemming
    #    print("\nERROR: Not enough or too many input arguments.\n")
    
start = time.time() #measure time taken
main()
print("time: ", time.time()-start, " sec")