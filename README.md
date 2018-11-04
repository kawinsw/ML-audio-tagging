# Machine Learning Challenge - Audio Tagging From Team "Life is Hard"

## Submission Details

(1) Submission file is located in model/baseline_prediction.csv
(2) Documented Model is in model/model-i where i refers to the i-th tag and the models are xgboost models. The parameters used to reach the model state are max_depth:5, eta:0.34, colsample_bytree:0.8, confidence_threshold:0.8, boost_rounds:5.
(2) Training and Inference Scripts are located in src/train.py. To reproduce the result, please follow the instructions at the bottom.

## Datasets
    
    audio files for all videos -- *.wav (https://drive.google.com/drive/folders/1w_DIUk9QNJcxex5DRaPD__d2t-Zae-Zs?usp=sharing)
    subtitles-train) -- data/subtitle_train.json
    subtitles-test) -- data/subtitle_test.json
    tags-train -- data/tags_train.csv
    tag dictionary -- data/tags_dict.json 

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
        
## How to evaluate GAP for your prediction 
```bash
python eval.py
```
* input: prediction, actual (same file format as for submission file) 
* output: GAP 