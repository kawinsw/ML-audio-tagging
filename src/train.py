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
np.random.seed(8423) 
import pandas as pd
import xgboost as xgb
import average_precision_calculator

NUM_PREPROCESS = "04"
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
    npdtest = np.load(os.path.join(prep_dir, "npdtest.npy"))
    train_y = np.load(os.path.join(prep_dir, "train_y.npy"))
    test_y = np.load(os.path.join(prep_dir, "test_y.npy"))
    
    dtest = xgb.DMatrix(npdtest)
    


    ## xgboost classifier
    print("Training...")
    test_num = 0
    
    res = np.zeros((9,9))
    
    ## NOTE: INSERT FOR LOOP
    for l in range(1, 10):
        for j in range(1, 10):
        
            # live spliting of train and test data
            X_train, X_val, y_train, y_val = train_test_split(npdtrain, train_y, test_size=0.20)
            dval = xgb.DMatrix(X_val)
            
            pred_test_dir = os.path.join('../', 'output', OUTPUT_PREFIX + str(test_num))
            test_num += 1

            params = {'max_depth':5, 'eta':0.34, 'colsample_bytree':0.8, 'silent': 1, 'booster':'gbtree', 'objective':'binary:logistic', 'min-child-weight':l}
            our_params = {'confidence_threshold': 0.8}
            #print("For {}, Our params:".format(OUTPUT_PREFIX + str(test_num)), params, our_params)

            pred_val = []
            pred_test = []
            modellist = []
            
            for i in range(NUM_CLASS):    
                ## training
                dtrain = xgb.DMatrix(X_train, y_train[:,i])
                clf = xgb.train(params, dtrain, num_boost_round=6 + l)
                modellist.append(clf)
                
                ## validation
                pred1 = clf.predict(dval)
                pred_val.append(pred1)
                ## test
                pred2 = clf.predict(dtest)
                pred_test.append(pred2)

            # Validate data
            pred_val = pd.DataFrame(np.array(pred_val).transpose())
            pred_val['LabelConfidencePairs'] = pred_val.iloc[:, 0:22].apply(lambda x: label_conf_pair(x, threshold=our_params['confidence_threshold']), axis=1)
            pred_val['Labels'] = pd.DataFrame(y_val).apply(lambda x: label_conf_pair(x, threshold=0.5), axis=1)
            pred_val = pred_val[['LabelConfidencePairs', 'Labels']]
            score = gap(pred_val)
            res[l - 1, j - 1] = score
            
            
            if score > 0.9:
                os.mkdir(pred_test_dir)
                for i, mod in enumerate(modellist):
                    mod.save_model(os.path.join(pred_test_dir, 'model-'+ str(i)))
                    
                #print('Validation GAP score: {}'.format(gap(pred_val)))

                # Create submission file - IGNORE ANYTHING BELONG
                print('Creating Submission File...')
                pred_test = pd.DataFrame(np.array(pred_test).transpose())
                pred_test['AudioId'] = label_test['AudioId']    
                pred_test = pd.merge(label_test, pred_test, on = 'AudioId')
                pred_test['Labels'] = pred_test['LabelConfidencePairs']
                pred_test['LabelConfidencePairs'] = pred_test.iloc[:,2:24].apply(lambda x: label_conf_pair(x, threshold=our_params['confidence_threshold']), axis=1)
                pred_test = pred_test[['AudioId','LabelConfidencePairs']]
                submission_output = os.path.join(pred_test_dir, 'baseline_prediction.csv')
                pred_test.to_csv(submission_output, index=False)
    
    print(res)
if __name__ == "__main__":
    main()
