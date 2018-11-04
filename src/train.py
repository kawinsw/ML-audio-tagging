import json 
from math import log
import re
import time
import gc
import urllib.request
import zipfile
import os
import argparse

from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
np.random.seed(1111) 
import pandas as pd
import xgboost as xgb

def decontracted(phrase): # to be fixed for all cases 
    ## specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    ## general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""
    ## Find the best match for the i first characters, assuming cost has
    ## been built for the i-1 first characters.
    ## Returns a pair (match_cost, match_length).
    words = open("./../data/words_by_frequency.txt").read().split()
    wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
    maxword = max(len(x) for x in words)
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return ' '.join(reversed(out))

def rawtext_to_words(raw_text,remove_stopwords=True):
    ## remove HTML
    text = bs(raw_text, 'lxml').get_text()
    ## remove non-letter
    text = re.sub('[^a-zA-Z]',' ', text)
    ## remove stopwords
    words = text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words('english')) 
        words = [w for w in words if not w in stops]
    return ' '.join(words)

def loadGloveModel(gloveFile):
    print ("Loading GloVe Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

def label_conf_pair(x, k = 5):
    return ' '.join(["{} {:.4f}".format(a_, b_) for a_, b_ in zip(x.sort_values(ascending=False).index.values[:k], x.sort_values(ascending=False).values[:k])])

def main():

    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_tr_dir", type=str, help="training subtitles, a json file", default = './../data/subtitle_train.json')
    parser.add_argument("--sub_te_dir", type=str, help="testing subtitles, a json file", default = './../data/subtitle_test.json')
    parser.add_argument("--label_train_dir", type=str, help="training tags, a csv file", default = './../data/tags_train.csv')
    parser.add_argument("--label_test_dir", type=str, help="sample submission file, a csv file", default = './../data/tags_test.csv')
    parser.add_argument("--pred_test_dir", type=str, help="submission file to save, a csv file", default = './../data/baseline_prediction.csv')

    args = parser.parse_args()
    sub_tr_dir = args.sub_tr_dir
    sub_te_dir = args.sub_te_dir
    label_train_dir = args.label_train_dir
    label_test_dir = args.label_test_dir
    pred_test_dir = args.pred_test_dir    

    ## load subtitles
    with open(sub_tr_dir) as f:
        sub_tr = json.load(f)
    with open(sub_te_dir) as f:
        sub_te = json.load(f)
    subtitles = {**sub_tr,**sub_te}
    
    print("Preprocessing...")
    corpus_set = []
    for key, value in subtitles.items():
        value = rawtext_to_words(decontracted(value))
        value = infer_spaces(''.join(value.split()))
        subtitles[key] = value
        corpus_set.append(value) 
    
    ## bag-of-word feature
    vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=1000)
    data_features = vectorizer.fit_transform(corpus_set)
    data_features = data_features.toarray()
    data_features.shape
    
    ## tfidf matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus_set) 
    word_lst = vectorizer.vocabulary_
    
    ## get word_embedding_matrix using GloVe
    if os.path.isfile('./../data/glove.42B.300d.txt'):
        pass
    else:
        urllib.request.urlretrieve("https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip", filename="./../data/glove.42B.300d.zip")
        with zipfile.ZipFile('./../data/glove.42B.300d.zip', 'r') as z:
            z.extractall('./../data/')
    
    glove_dic = loadGloveModel('../data/glove.42B.300d.txt')
    glove_matrix = []
    invalid_lst = {}
    for i, word in enumerate(word_lst):
        try:
            glove_matrix.append(glove_dic[word])
        except:
            invalid_lst[i] = word
    
    ## tfidf-GloVe feature 
    tfidf_matrix_deleted = np.delete(tfidf_matrix.todense().T, tuple(list(invalid_lst)), axis = 0)
    text_glove_matrix = np.matmul(tfidf_matrix_deleted.T, np.array(glove_matrix))   
    
    ## prepare target label
    label_train = pd.read_csv(label_train_dir)
    label_test = pd.read_csv(label_test_dir)
    labels = pd.concat([label_train,label_test]) 
    labels['LabelConfidencePairs'] = labels['LabelConfidencePairs'].apply(lambda x: x.split()[0::2])
    label_encoded = labels['LabelConfidencePairs'].str.join('|').str.get_dummies()
    label_encoded.columns = list(map( int , label_encoded.columns ))
    label_encoded = label_encoded.reindex(sorted(label_encoded.columns), axis=1)
    label_encoded = np.array(label_encoded).astype(float) 
    train_y = label_encoded[:label_train.shape[0],]
    test_y = label_encoded[label_train.shape[0]:,]
    
    ## xgboost classifier
    print("Training...")
    params = {'max_depth':6, 'eta':0.4, 'colsample_bytree':0.3, 'silent': 1, 'booster':'gbtree', 'objective':'binary:logistic'}
    
    data_features  = np.concatenate((data_features,text_glove_matrix), axis=1)
    dtest = xgb.DMatrix(data_features[label_train.shape[0]:,:])
    pred_test = []
    num_class = 22
    for i in range(num_class): 
        print('Tag {}'.format(i))
        ## training
        dtrain = xgb.DMatrix(data_features[:label_train.shape[0],:], train_y[:,i])
        clf = xgb.train(params, dtrain, num_boost_round=15)
        ## test
        pred = clf.predict(dtest)
        pred_test.append(pred)
        
    print('Creating Submission File...')
    pred_test = pd.DataFrame(np.array(pred_test).transpose())
    pred_test['AudioId'] = label_test['AudioId']    
    pred_test = pd.merge(label_test, pred_test, on = 'AudioId')
    pred_test['Labels'] = pred_test['LabelConfidencePairs'] 
    pred_test['LabelConfidencePairs'] = pred_test.iloc[:,2:24].apply(label_conf_pair, axis=1)
    pred_test = pred_test[['AudioId','LabelConfidencePairs']]
    pred_test.to_csv(pred_test_dir, index=False)

    print('###### Run time: %d seconds.' %(time.time()-start))
    
if __name__ == "__main__":
    main()
