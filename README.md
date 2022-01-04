---

<div align="center">    
 
# Bill Prediction with Sentence-BERT 🚀⚡🔥
### By: *August Lohse, S216350; Espen Rostrup, S215937; Matias Piqueras, S216005*
 
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

To generate and run most outputs and models, you will have to download the embedding tensors from sentence-BERT (including a finetuned version) as these are to big to store on Github (links below). Place the tensors in the directory `/data/processed/` .
* [sentence-BERT embeddings](https://drive.google.com/drive/folders/1K5EI0axL9OyrCGi6s0Ivd7bDHuVtgTZL?usp=sharing)

If you just want to replicate the plots and tables presented in the paper then
```bash
# src folder 
cd make_plot 
```
and run  `baseline_models_and_plots.py` which loads the trained models from the directory `/trained_models`. It prints metrics to console and creates plots and tables in `/plots_tables` 

If you instead want to train the models then you can run the following commands
```bash
# module folder
python3 -m src.train_sbert_downstream
```
Where the flag `--finetuned_embeddings` indicates if the finetuned embeddings should be used or not. The standard BERT can be trained using the notebook `finetuning-BERT.ipynb`.

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

A python script has been prepared for finetuning sentence BERT. It can be found in `/src/prepare_data/fine-tuning_SBERT.py`
The fine-tuned model is stored locally, when running the script. It will output validation metrics each epoch. 
We have made our final fine-tuned model accesible through [Google Drive](https://drive.google.com/drive/folders/1og5VgL5DlmwxzBnCnGKRXbpRmowSSbK5?usp=sharing). In the zip-file their is a README explaining how to use the model. 

**4. Extracting Bill Embeddings**

To extract the Bill Embeddings we feed to the downstream tasks we pass the data prepared in step 2 to Sentence-BERT.  

**Extra: getting reuslts for plots and tables**

If you wish to train new models, and obtain create new results, plots and tables, prepare the data as described, then: 

**5. Train models**

Place them in `/trained models´ - make sure that they are named with "meta" and "CNN" or "FNN" as well as "avg" if you average the the sentence embeddings in the FNN. This will make sure that the models are loaded correctly in the next step.

**6. Predict on data**

Run make_predictions.py in /prepare_data.py - This will create a predictions.pkl file in the data/results folder. This file contains a dictionary with all the model names as keys, and contains targets, predicted, probas and false/negative positive rate as well as precision recall curve. This file is used for plotting and creating tables



## References

> Kornilova, A., & Eidelman, V. (2019). Billsum: A corpus for automatic summarization of us legislation. arXiv preprint arXiv:1910.00523.
