---

<div align="center">    
 
# Bill Prediction with Sentence-BERT    
 
</div>
 
## Description   
Here we write a short project description.

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

Several modules under `/src` are used to prepare the data for our models. This includes data cleaning, finetuning both sentence-BERT and vanilla BERT and extracting document embeddings. 


