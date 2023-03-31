import csv
import sys
import nltk
from nltk.corpus import stopwords
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

def cleanText(text, switch): #clean text before partitioning based on command line input
    if switch == 1:
        clean_text = preprocessStop(preprocessLC(text))
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
    if real_value =='positive':
        if classified_value == 'positive':
            mat_1[1] += 1
        elif classified_value == 'negative':
            mat_1[2] += 1
        else:
            mat_1[3] += 1
    elif real_value == 'negative':
        if classified_value == 'positive':
            mat_2[1] += 1
        elif classified_value == 'negative':
            mat_2[2] += 1
        else:
            mat_2[3] += 1
    else:
        if classified_value == 'positive':
            mat_3[1] += 1
        elif classified_value == 'negative':
            mat_3[2] += 1
        else:
            mat_3[3] += 1

def sentence_input(prob_pos_word, prob_neg_word, prob_neu_word, prob_pos, prob_neg, prob_neu, vocabulary, switch):
    userInput = (input("Enter your sentence: "))
    print(f"\n Sentence S: \n\n{userInput}\n")
    cleaned = cleanText(userInput, switch)
    is_pos = math.log(prob_pos)
    is_neg = math.log(prob_neg)
    is_neu = math.log(prob_neu)
    for j in range(len(cleaned)): #positive probability
        if cleaned[j] in vocabulary:
            is_pos = is_pos + prob_pos_word[cleaned[j]]
    for j in range(len(cleaned)): #negative probability
        if cleaned[j] in vocabulary:
            is_neg = is_neg + prob_neg_word[cleaned[j]]
    for j in range(len(cleaned)): #neutral probability
        if cleaned[j] in vocabulary:
            is_neu = is_neu + prob_neu_word[cleaned[j]]
    if max(is_pos, is_neg, is_neu) == is_pos:# classification based on trained set
        classification = "positive"
    elif max(is_pos, is_neg, is_neu) == is_neg:
        classification = "negative"
    else:
        classification = "neutral"
    print(f"was classified as {classification}.")
    print(f"P(positive | S) = {math.exp(is_pos)}")
    print(f"P(negative | S) = {math.exp(is_neg)}")
    print(f"P(neutral | S) = {math.exp(is_neu)}")
    userInput = (input("\nDo you want to enter another sentence [Y/N]?"))
    if userInput == 'Y':
        sentence_input(prob_pos_word, prob_neg_word, prob_neu_word, prob_pos, prob_neg, prob_neu, vocabulary, switch)
    elif userInput == 'N':
        return
    else:
        sentence_input(prob_pos_word, prob_neg_word, prob_neu_word, prob_pos, prob_neg, prob_neu, vocabulary, switch)

def main():
    n = int((len(t)-1)*80/100)
    boundary = int((len(t)-1)*80/100)
    train_set = [] #train set list
    test_set = [] #test set list
    pos = []
    neu = []
    neg = []

    mat_0 = ["real\predicted", "positive", "negative", "neutral"]
    mat_1 = ["positive", 0, 0, 0]
    mat_2 = ["negative", 0, 0, 0]
    mat_3 = ["neutral", 0, 0, 0]

    switch = 0 #check argument to pass preprocess or not
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        if inp == "YES":
            switch = 1

#NAME, A NUMBER PREPROCESSING            
    print("\nJungwoo, Hwang, A20478312 solution:")
    if switch == 1:
        print("Ignored pre-processing step: STEMMING\n")
    else:
        print("Ignored pre-processing step: \n")

    print("Training classifier...")
    for i in range(1, n):
        cleaned_train = cleanText(t[i][10], switch)
        train_set.append(cleaned_train)
    vocabulary = vocabb(train_set)
#voca preprocess done
#    print("voca done\n")

    for i in range(len(train_set)):
        if t[i][1] == 'positive': #2003
            pos.append(train_set[i])
        elif t[i][1] == 'negative': #7090
            neg.append(train_set[i])
        else: #2618
            neu.append(train_set[i])
#classify done
#    print("train set: ", len(train_set))
#    print("positive: ", len(pos))
#    print("negative: ", len(neg))
#    print("neutral: ", len(neu))
#    print("classification done\n")

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
#    print("bag of words done\n")

    for i in range(len(vocabulary)): # store probabilities
        prob_pos_word[vocabulary[i]] = (positive(vocabulary[i], vocabulary, pos_bag_of_words, pos_count))
        prob_neg_word[vocabulary[i]] = (negative(vocabulary[i], vocabulary, neg_bag_of_words, neg_count))
        prob_neu_word[vocabulary[i]] = (neutral(vocabulary[i], vocabulary, neu_bag_of_words, neu_count))
#probabilty of word done
#    print("probability done\ntraining done\n")

    print("Testing classifier...")

    for i in range(boundary+1, len(t)):
        cleaned_test = cleanText(t[i][10], switch)
        test_set.append(cleaned_test)
#test set
#    print("test set preprocess done")
#    print("test set: ", len(test_set))
#    print()


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
    print("Test results / metircs: ")
    print(mat_0)
    print(mat_1)
    print(mat_2)
    print(mat_3)

    pos_tp = mat_1[1]
    pos_tn = mat_2[2]+mat_2[3]+mat_3[2]+mat_3[3]
    pos_fp = mat_2[1]+mat_3[1]
    pos_fn = mat_1[2]+mat_1[3]
    pos_sen = pos_tp/(pos_tp+pos_fn)
    pos_spe = pos_tn/(pos_tn+pos_fp)
    pos_pre = pos_tp/(pos_tp+pos_fp)
    pos_pv = pos_tn/(pos_tn+pos_fn)
    pos_acc = (pos_tp+pos_tn)/(pos_tp+pos_tn+pos_fp+pos_fn)
    pos_fs = pos_tp/(pos_tp+0.5*(pos_fp+pos_fn))

    neg_tp = mat_2[2]
    neg_tn = mat_1[1]+mat_1[3]+mat_3[1]+mat_3[3]
    neg_fp = mat_1[2]+mat_3[2]
    neg_fn = mat_2[1]+mat_2[3]
    neg_sen = neg_tp/(neg_tp+neg_fn)
    neg_spe = neg_tn/(neg_tn+neg_fp)
    neg_pre = neg_tp/(neg_tp+neg_fp)
    neg_pv = neg_tn/(neg_tn+neg_fn)
    neg_acc = (neg_tp+neg_tn)/(neg_tp+neg_tn+neg_fp+neg_fn)
    neg_fs = neg_tp/(neg_tp+0.5*(neg_fp+neg_fn))

    neu_tp = mat_3[3]
    neu_tn = mat_1[1]+mat_1[2]+mat_2[1]+mat_2[2]
    neu_fp = mat_1[3]+mat_2[3]
    neu_fn = mat_3[1]+mat_3[2]
    neu_sen = neu_tp/(neu_tp+neu_fn)
    neu_spe = neu_tn/(neu_tn+neu_fp)
    neu_pre = neu_tp/(neu_tp+neu_fp)
    neu_pv = neu_tn/(neu_tn+neu_fn)
    neu_acc = (neu_tp+neu_tn)/(neu_tp+neu_tn+neu_fp+neu_fn)
    neu_fs = neu_tp/(neu_tp+0.5*(neu_fp+neu_fn))
    
    print("\n   Positive / Negative, Neutral")
    print(f"Number of true positives: {pos_tp}")
    print(f"Number of true negatives: {pos_tn}")
    print(f"Number of false positives: {pos_fp}")
    print(f"Number of false negatives: {pos_fn}")
    print(f"Sensitivity: {pos_sen}")
    print(f"Specificity: {pos_spe}")
    print(f"Precision: {pos_pre}")
    print(f"Negative predictive value: {pos_pv}")
    print(f"Accuracy: {pos_acc}")
    print(f"F-score: {pos_fs}")

    print("\n   Negative / Positive, Neutral")
    print(f"Number of true positives: {neg_tp}")
    print(f"Number of true negatives: {neg_tn}")
    print(f"Number of false positives: {neg_fp}")
    print(f"Number of false negatives: {neg_fn}")
    print(f"Sensitivity: {neg_sen}")
    print(f"Specificity: {neg_spe}")
    print(f"Precision: {neg_pre}")
    print(f"Negative predictive value: {neg_pv}")
    print(f"Accuracy: {neg_acc}")
    print(f"F-score: {neg_fs}")
    
    print("\n   Neutral / Positive, Negative")
    print(f"Number of true positives: {neu_tp}")
    print(f"Number of true negatives: {neu_tn}")
    print(f"Number of false positives: {neu_fp}")
    print(f"Number of false negatives: {neu_fn}")
    print(f"Sensitivity: {neu_sen}")
    print(f"Specificity: {neu_spe}")
    print(f"Precision: {neu_pre}")
    print(f"Negative predictive value: {neu_pv}")
    print(f"Accuracy: {neu_acc}")
    print(f"F-score: {neu_fs}\n")

    sentence_input(prob_pos_word, prob_neg_word, prob_neu_word, prob_pos, prob_neg, prob_neu, vocabulary, switch)

    
#start = time.time() #measure time taken
main()
#print("\ntime: ", time.time()-start, "sec")
