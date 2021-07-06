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