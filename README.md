# Dependencies
* Python 3.10

# Installation
To install the required packages, run the following command:
```pip
pip install -r requirements.txt
```
This command will install the necessary Python packages specified in the requirements.txt file.

# Usage
## 1. Download the Dataset
Download the CelebA dataset from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html "link") and extract it to the desired location.

Unzip the "img_align_celeba.zip" file and move the extracted contents to the "data/img_align_celeba" directory.

## 2. Preprocess the Dataset
To preprocess the dataset, run the following command:
```python
python preprocess.py
```
## 3. Train the Model
To train the face classification model, run the following command:
```python
python train.py
```

