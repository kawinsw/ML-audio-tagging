import argparse
import pandas as pd
import average_precision_calculator

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, help="prediction file, a csv file", default = '../data/baseline_prediction.csv')
    parser.add_argument("--y_file", type=str, help="ground truth file, a csv file", default = '../data/tags_test.csv')
 
    args = parser.parse_args()
    pred_file = args.pred_file
    y_file = args.y_file

    pred = pd.read_csv(pred_file)
    pred = pred[['AudioId','LabelConfidencePairs']]
    y = pd.read_csv(y_file)
    y.columns = ['AudioId', 'Labels']
    combined = pred.merge(y, on='AudioId')

    print('GAP: {}'.format(gap(combined)))
