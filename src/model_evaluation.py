import json 
from math import log
import re
import time
import gc
import urllib.request
import zipfile
import os
import argparse

from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(11235) 
import pandas as pd
import xgboost as xgb
import average_precision_calculator

NUM_PREPROCESS = "01"
OUTPUT_PREFIX = "test"

# DO NOT CHANGE THE FOLLOWING VAR
NUM_CLASS = 22

def process(label):
    label_new = []
    label = label.split()
    for i in range(0,len(label),2):
        label_new.append(label[i])
    return label_new

def gap(merge_file):
    conf = []
    pred = []
    label = []
    true = []
    for i in range(len(merge_file)):
        pred_p = merge_file.iloc[i]['LabelConfidencePairs']
        pred_p = pred_p.split()
        labels = process(merge_file.iloc[i]['Labels'])
        for a in range(0, len(pred_p),2):
            if pred_p[a] in labels:
                conf.append(float(pred_p[(a+1)]))
                pred.append(pred_p[a])
                label.append(merge_file.iloc[i]['Labels'])
                true.append(1)
            else:
                conf.append(float(pred_p[(a+1)]))
                pred.append(pred_p[a])
                label.append(merge_file.iloc[i]['Labels'])
                true.append(0)

    x = pd.DataFrame({'pred': pred, 'conf': conf, 'label':label, 'true': true})
    x = x.sort_values(by = 'conf', ascending = False)
    p = x.conf.values
    a = x.true.values
    ap = average_precision_calculator.AveragePrecisionCalculator.ap(p, a)
    return ap

def label_conf_pair(x, k = 5, threshold=False):
    if threshold != False:
        k = min(k, (x > threshold).sum())
        k = max(k, 1)
    label = ' '.join(["{} {:.4f}".format(a_, b_) for a_, b_ in zip(x.sort_values(ascending=False).index.values[:k], x.sort_values(ascending=False).values[:k])])
    return label

def main():
    print('Loading...')

    label_test_dir = './../data/original/tags_test.csv'
    label_test = pd.read_csv(label_test_dir)

    prep_dir = os.path.join("../", "data", "prep" + NUM_PREPROCESS)
    
    npdtrain = np.load(os.path.join(prep_dir, "npdtrain.npy"))
    train_y = np.load(os.path.join(prep_dir, "train_y.npy"))
    
    dval = xgb.DMatrix(npdtrain)

    ## xgboost classifier
    print("Evaluating...")

    pred_val = []
    pred_test = []

    our_params = {'confidence_threshold': 0.5}
    
    for i in range(NUM_CLASS):
        ## validation
        clf = xgb.Booster({'nthread': 4}) 
        clf.load_model('../output/test26/model-' + str(i))
        pred1 = clf.predict(dval)
        pred_val.append(pred1)
        ## test
        pred2 = clf.predict(dtest)
        pred_test.append(pred2)

    # Validate data
    pred_val = pd.DataFrame(np.array(pred_val).transpose())
    pred_val['LabelConfidencePairs'] = pred_val.iloc[:, 0:22].apply(lambda x: label_conf_pair(x, threshold=our_params['confidence_threshold']), axis=1)
    pred_val['Labels'] = pd.DataFrame(train_y).apply(lambda x: label_conf_pair(x, threshold=0.5), axis=1)
    pred_val = pred_val[['LabelConfidencePairs', 'Labels']]
    score = gap(pred_val)
    print('Validation GAP score: {}'.format(gap(pred_val)))
    
if __name__ == "__main__":
    main()