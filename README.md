# Eluvio Machine Learning Challenge - Audio Tagging

## Datasets
    
    audio files for all videos -- *.wav (https://drive.google.com/drive/folders/1w_DIUk9QNJcxex5DRaPD__d2t-Zae-Zs?usp=sharing)
    subtitles-train) -- data/subtitle_train.json
    subtitles-test) -- data/subtitle_test.json
    tags-train -- data/tags_train.csv
    tag dictionary -- data/tags_dict.json

## Baseline model based on subtitles only
    
To facilitate your modeling process, we provide a baseline model: a gradient boosting tree model trained on features of bag-of-words and tf-idf weighted word embedding (GloVe), using subtitles only. We also upload evaluation routines to help improve your model. 

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
python train.py
```
* input: subtitle_train.json, subtitle_test.json, tags_train.csv, tags_test.csv (constant label prediction)
* output: baseline_prediction.csv (submission file)
        
## How to evaluate GAP for your prediction 
```bash
python eval.py
```
* input: prediction, actual (same file format as for submission file) 
* output: GAP 
    
