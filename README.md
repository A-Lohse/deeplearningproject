---

<div align="center">    
 
# Bill Prediction with Sentence-BERT    
 
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
cd src
python3 train_sbert_downstream.py
python3 train_sbert_downstream.py --finetuned_embeddings
python3 train_vanilla_bert.py
python3 train_baseline.py??
```
Where the flag `--finetuned_embeddings` indicates if the finetuned embeddings should be used or not. 

### Extra

Several modules under `/src/prepare_data/` are used to prepare the data for our models. This includes data cleaning, finetuning both sentence-BERT and vanilla BERT and extracting document embeddings. Below follows an overview of what they do.

**1. Generating metadata**

**2. Generate finetuning data for BERT**

The data used to finetune BERT comes from the [BillSum](https://github.com/FiscalNote/BillSum) project. Specifically the two datafiles `data/raw/us_train_sent_scores.pkl` and `data/raw/us_train_sent_scores.pkl` are used. The module `generate_bert_finetuning_data.py` extracts the relevant text from BillSum data and matches it with bill with meta data, including if the Bill was enacted or not. 

**3. Finetuning sentence-BERT**


**4. Extracting Bill Embeddings**



## References

> Kornilova, A., & Eidelman, V. (2019). Billsum: A corpus for automatic summarization of us legislation. arXiv preprint arXiv:1910.00523.
