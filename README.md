# MKDTI


# MKDTI: Predicting drug-target interactions via multiple kernel fusion on graph attention network

![MKDTI](https://github.com/jackedv/MKDTI/blob/main/MKDTI.png?raw=true)

# Installation
MKDTI is based on Pytorch and Python
## Requirements
You will need the following packages to run the code:
* python==3.8.16
* torch==1.12.1
* numpy==1.23.5
* pandas==1.5.3
* torch-cluster==1.6.0
* torch-geometric==2.2.0 
* torch-scatter==2.1.0
* torch-sparse==0.6.16
* torch-spline-conv==1.2.1
* torchaudio==0.12.1
* torchvision==0.13.1
* pytorch-mutex==1.0

# Data Description
The './data' folder contains the data used by our paper. 
Includes drug information, target information; drug similarity matrix, target similarity matrix, drug-target association matrix
# Usage
First, you need to clone the repository or download source codes and data files. 

    $ git clone https://github.com/jackedv/MKDTI.git

Then go to the folder '/code'

    $ cd code

You can directly run the following code to train the model:
  
    python main.py