

1.dataset
Download dataset.zip from https://drive.google.com/file/d/1YHUU1lMay2dqBiO0bjwg_aKXk9tJvhWb/view?usp=share_link and unzip it into anywhere, and set path of training set and test set in config/main.yaml: */TrainDataset and */TestDataset

2.test
Download saved_modelfile.zip from https://drive.google.com/file/d/1ruOtgBbSZUvKSbJFl5ViSzx6EEg_6Zp8/view?usp=share_link and unzip it into main directory.

Let pretrained weights in "UPformer/saved_modelfile/indus1000-new/*"

Run "python test.py" to test the pretrained models for best results on different subsets.

3.train
Run "python train.py" to train the model for best results on different subsets.


Thanks:
This code is inspired by: https://github.com/fanyang587/UGTR and https://github.com/cvlab-yonsei/MNAD