# 目前進度
### Face classification
* train.py 有更新，目前已經train好了(20000 photos, batch size = 64, epochs = 3)，下面可以下載
* test.py 已測試完成
### Music recommendation
* None

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
Download the CelebA dataset from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html "link").

Unzip the "img_align_celeba.zip" file and move the extracted contents to the "data/img_align_celeba" directory.

Download and move the "list_attr_celeba.txt" file into the "data" folder.

## 2. Preprocess the Dataset
To preprocess the dataset, run the following command:
```python
python preprocess.py
```
You can limit the number of photos by adjusting the "num_photos" parameter.

## 3. Train the Model
To train the face classification model, run the following command:
```python
python train.py
```
You can download the trained model from [here](https://drive.google.com/drive/folders/146qbJXDoewV6p73qA4vFUPLgEmL7svVi?usp=drive_link) if you want it.

## 4. Test the model
To test the model, put the photo in the "input" folder and run following command:
```python
python image_process.py
```
```python
python test.py
```


