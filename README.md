# BDCNN: An Exploration of Bi-Directional Convolutional Neural Network Training for Image Classification

### Introduction

This repo is based on the methods of our paper. In our paper, we present a combination of bi-directional training and convolutional models. In this repo, we implement conventional CNN, FCN and their bi-directional variants, BDCNN and BDFCN. The results we obtained indicates that our bi-directional models can improve the image classification performance comparing with the base models.

The dataset we used for our experiment is from VehicleX([Paper](https://arxiv.org/abs/1912.08855), [Github Repo](https://github.com/yorkeyao/VehicleX)), a large-scale synthetic vehicle image dataset. In our experiment, we trained our models as classifiers to predict the vehicle type (11 in total) of a given grayscale image with size of 64 by 64.

### Model Training

In this repo, we implemented our models in Pytorch, and the model implementations are in [model.py](https://github.com/camerontan0527/bdcnn/blob/main/model.py). The file [data_loader.py](https://github.com/camerontan0527/bdcnn/blob/main/data_loader.py) is used to load preprocessed data for training and testing. The base models and their bi-directional variants are trained in [train.py](https://github.com/camerontan0527/bdcnn/blob/main/train.py) with 3 optional arguments, which are lr (learning rate), epoch (number of epochs to train) and model (selected model to train with 2 options: fcn and cnn). In the training file, the base model would firstly be trained with the fixed number of epochs, and then trained the bi-directional model for the fixed number of epochs. Note that the bi-directional models are simulated by the combination of a model in forward training flow and a model in reversed training flow, and the corresponding parameters of the 2 models are maintained to be equivalent constantly, since several pairs of parameters in the 2 models of different flow direction stand for the same parameter in conventional bi-directional training.

An example of training our model would be using command line operation like following:
```
python3 train.py --lr 0.01 --epoch 100 --model fcn
```
Then a conventional FCN and its bi-directional variants would be trained for 100 epochs with a learning rate of 0.01.

### About Dataset

Due to privacy protocol, the dataset we used in our experiments would not be shared publicly. Instead, to show the format of the data for training, we add some dummy input and target under the directory [data](https://github.com/camerontan0527/bdcnn/tree/main/data). In the provided dummy dataset, {train, test, val}_data is the input for the model, where there are 11, 1 and 1 instances for training, testing and validating, respectively. The corresponding target (class label) for each instance is recorded in {train, test, val}_type files.

### Train for Custom Dataset
We also uploaded a data preprocessing file, [data_preprocess.py](https://github.com/camerontan0527/bdcnn/blob/main/data_preprocess.py), to convert raw images to desired format for training. To use the code we present for preprocessing, please split your image set to train, test and validation sets, and then put the images of each set under the direcotry 'data/{train, test, val}', respectively. The preprocess file would convert all images under each set to grayscale with size of 64 by 64, then save all image data of each set to a numpy array and save as {train, test, val}_data.npy under [data](https://github.com/camerontan0527/bdcnn/tree/main/data) directory. Both RGB and grayscale images would be accepted for preprocessing, and please ensure that the pixel values of all images are in the range of [0, 255].

Unfortunately, we did not provide code for preprocessing training targets. Please label the class of each instance in your dataset and save them as pytorch tensor files (.pth) for your train, test and validation set. Additionally, make sure that the $n$th instance in {train, test, val}_data.npy match with the $n$th target in your generated target files. The target files should be named as '{train, test, val}_type.pth' under [data](https://github.com/camerontan0527/bdcnn/tree/main/data) directory. Then the implemented models would be ready to be trained.