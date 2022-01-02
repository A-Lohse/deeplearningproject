---

<div align="center">    
 
# Bill Prediction with Sentence-BERT    
 
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/A-Lohse/deeplearningproject
# install project   
cd deeplearningproject
pip install -r requirements.txt
 ```   

To run most models, you will have to download the embedding tensors from sentence-BERT (both vanilla and finetuned) as these are to big to store on Github. They can be found here:

* [sentence-BERT embddings](gdrive)
* [finetuned sentence-BERT embddings](gdrive)



```bash
# module folder
cd project
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```
