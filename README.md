# Fact Extraction and VERification

This practical implements the baseline from [FEVER: A large-scale dataset for Fact Extraction and VERification.](https://arxiv.org/abs/1803.05355) and an improved system  from [GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification](https://arxiv.org/abs/1908.01843).


## Installation

Clone the repositor

Install requirements 

    pip install -r requirements.txt

## Data Preparation

### 1. download dataset  
Download the FEVER dataset from the [website](http://fever.ai/data.html) into the data directory
    
    # We use the data used in the baseline paper
    wget -O baseline/data/fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
    wget -O baseline/data/fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl
    wget -O baseline/data/fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl
    
I adopt the code from [ACL 2019 paper](https://github.com/thunlp/GEAR).

The evidence extraction results can be found in [Google Cloud](https://drive.google.com/drive/folders/1y-5VdcrqEEMtU8zIGcREacN1JCHqSp5K).

Download these three files and put them in the ``data/retrieved/`` folder. 

### 2. download the database
```
# Download the fever database
wget -O data/fever/fever.db https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db
```

### For baseline system
Create a term-document count matrix for each split, and then merge the count matrices.
    
    wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
    unzip wiki-pages.zip -d baseline/data
    python build_db.py data/wiki-pages baseline/data/fever --num-files 5
    python build_count_matrix.py baseline/data/fever baseline/data/index
    python merge_count_matrix.py baseline/data/index baseline/data/index

Use TF-IDF here(the default setting is bigram and unigram combined)
    
    python reweight_count_matrix.py baseline/data/index/count-ngram\=2-hash\=16777216.npz baseline/data/index --model tfidf
  

### For improved system
Extract evidence sentences.

```
# Extract the evidence from database
cd scripts/
python retrieval_to_bert_input.py

# Build the datasets for gear
python build_gear_input_set.py

cd ..
```

Feature extraction needs to download our pre-trained BERT-Pair model ([Google Cloud](https://drive.google.com/drive/folders/1y-5VdcrqEEMtU8zIGcREacN1JCHqSp5K)) and put the files into the ``pretrained_models/BERT-Pair/`` folder.

Then the folder will look like this:
```
pretrained_models/BERT-pair/
    	pytorch_model.bin
    	vocab.txt
    	bert_config.json
```

Then run the feature extraction scripts. This procedure would take a long time.
```
cd feature_extractor/
chmod +x *.sh
./train_extracor.sh
./dev_extractor.sh
./test_extractor.sh
cd ..
```

## Training

The remaining tasks of the baseline are done in Jupyter notebook `fever.ipynb`. 

For the improved system the tasks are done as follows.
```
#training
cd
CUDA_VISIBLE_DEVICES=0 python train.py
```

```
#testing
CUDA_VISIBLE_DEVICES=0 python test.py
```

```
#result output
python results_scorer.py
cd ..
```

