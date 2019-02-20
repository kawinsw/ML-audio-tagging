# Eluvio Machine Learning Challenge - *Tigger Tagger* Audio Tagging
Team members: Kawin Swaddiwudhipong, Seah Ying Hang, Huang Yueh Han, Tommy Poa


## Submission Details

1. Submission file is best_submission.csv
2. Documented Model is in model/model-i where i refers to the i-th tag and the models are xgboost models. The parameters used to reach the model state are max_depth:5, eta:0.34, colsample_bytree:0.8, confidence_threshold:0.8, boost_rounds:5.
3. Training and Inference Scripts are located in src/train.py. To reproduce the result, please follow the instructions at the bottom.

## Datasets
    
    audio files for all videos -- *.wav (https://drive.google.com/drive/folders/1w_DIUk9QNJcxex5DRaPD__d2t-Zae-Zs?usp=sharing)
    subtitles-train) -- data/subtitle_train.json
    subtitles-test) -- data/subtitle_test.json
    tags-train -- data/tags_train.csv
    tag dictionary -- data/tags_dict.json 

## Description

Machine learning solution for end-to-end audio classification/labelling, building towards a universal video classification machine to better understand various types of videos. 

Available data was split randomly into training and validation sets in an 80:20 ratio, which were preprocessed and cached. The training script sweeps over a range of hyperparameters to train models, before evaluating them based on their Global Average Precision (GAP) score for the validation dataset. Those that exceeded a performance threshold are then saved in output.

* Created a framework to split training data to evaluate and optimize the performance of a gradient boosting tree (xgboost) model trained on features of bag-of-words and tf-idf weighted word embedding (GloVe) using hyperparameter sweeping.

* Produced/Created training and inference scripts, a chosen documented model and a set of predictions for the blind test data. 
 
## Tools used

    numpy
    sklearn
    nltk
    pandas
    xgboost
    GloVe
    BeautifulSoup



## Install dependencies

### Using Anaconda (recommended)

1. Install [Anaconda (Python 3.7)](https://www.anaconda.com/download/#download)
2. Setup Python environment:
```bash
ENV_NAME=calhacks
make env-setup/$ENV_NAME
source activate $ENV_NAME

python -c "import nltk; nltk.download('stopwords')"
```

### Using pip directly (python 3.5+, pip3 required)
```bash 
pip3 install --no-cache-dir -r requirements.txt

python3 -c "import nltk; nltk.download('stopwords')"
```
    
## How to train
```bash
python preprocess.py
python train.py
```
* input: subtitle_train.json, subtitle_test.json, tags_train.csv, tags_test.csv (constant label prediction)
* output: output/exptfolder/*

We first preprocess the data and cache the values into data/prep{i}/ using numpy arrays. The training script will sweep through a range of hyperparameters to find the model with the best validation GAP score. The models that reach a certain level of performance are then saved in output. 
        
## Evaluation of GAP for a prediction 
```bash
python eval.py
```
* input: prediction, actual (same file format as for submission file) 
* output: GAP 
