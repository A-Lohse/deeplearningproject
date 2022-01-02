---

<div align="center">    
 
# Bill Prediction with Sentence-BERT 🚀⚡🔥
 
 
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
 
</div>
 
## Description   
This repository contains the code used for our project in the course **[02456 Deep Learning](https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch)** at the Technical University of Denmark (DTU).

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/A-Lohse/deeplearningproject
# install project   
cd deeplearningproject
pip install -r requirements.txt
 ```   

To generate and run most outputs and models, you will have to download the embedding tensors from sentence-BERT (including a finetuned version) as these are to big to store on Github (links below). Place the tensors in the directory `/data/processed/`.
* [sentence-BERT embeddings](gdrive)
* [finetuned sentence-BERT embeddings](gdrive)

If you just want to replicate the plots and tables presented in the paper then
```bash
# module folder
cd notebooks
```
and run the notebook `analysis.ipynb` which loads the trained models from the directory `/trained_models`. If you instead want to train the models then you can run the following commands
```bash
# module folder
python3 -m src.train_sbert_downstream
python3 -m src.train_sbert_downstream --finetuned_embeddings
python3 -m src.train_vanilla_bert
python3 -m src.train_baseline??
```
Where the flag `--finetuned_embeddings` indicates if the finetuned embeddings should be used or not. 

### Extra

Several modules under `/src/prepare_data/` are used to prepare the data for our models. This includes data cleaning, finetuning both sentence-BERT and vanilla BERT and extracting document embeddings. Below follows an overview of what they do.

**1. Generating metadata**

Apart from the Bill text we include the following metadata

* `bill_status (outcome variable)`: Dummy of bill status (1 if enacted, 0 otherwise)
* `cosponsors`: Interger value of the amount of cosponsors
* `majority`: Dummy of if bill proposing party is in majority
* `party`: Party dummy
* `gender`: Dummy of if the bill proposing politician is male/female

The data comes from the [Congressional Bills Project](http://congressionalbills.org/) and the original data can be downloaded [here](http://congressionalbills.org/download.html) and is prepared using the script `generate_metadata.py`. 

**2. Generate finetuning and embedding extraction data for BERT/S-BERT**

The Bill text data used to finetune BERT and extract bill Embeddings comes from the [BillSum](https://github.com/FiscalNote/BillSum) project. Specifically the two datafiles `data/raw/us_train_sent_scores.pkl` and `data/raw/us_train_sent_scores.pkl` are used. The module `generate_bert_finetuning_data.py` extracts the relevant text from BillSum data and merges it with the bill with meta data, including if the Bill was enacted or not through the unique bill ID. 

**3. Finetuning sentence-BERT**


**4. Extracting Bill Embeddings**

To extract the Bill Embeddings we feed to the downstream tasks we pass the data prepared in step 2 to Sentence-BERT.  


## References

> Kornilova, A., & Eidelman, V. (2019). Billsum: A corpus for automatic summarization of us legislation. arXiv preprint arXiv:1910.00523.
