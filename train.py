import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import *
from data_loader import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Add and read arguments
    par = argparse.ArgumentParser()
    par.add_argument('--lr', help='learning rate', default=1e-5)
    par.add_argument('--epoch', help='training epochs', default=100)
    par.add_argument('--model', help='types of model to train, should be either \'fcn\' or \'cnn\'', default='cnn')
    args = par.parse_args()
    lr = float(args.lr)
    n_epoch = int(args.epoch)
    model_type = args.model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    # All image data are converted to .npy files by train, validation and test set
    # Note that the pixel values in our image are in range [0, 255]
    train_img = np.load('./data/train_data.npy')
    test_img = np.load('./data/test_data.npy')
    val_img = np.load('./data/val_data.npy')

    # Targets (class label) are saved in .pth files
    train_typeID = torch.load('./data/train_type.pth')
    test_typeID = torch.load('./data/test_type.pth')
    val_typeID = torch.load('./data/val_type.pth')

    # Normalize pixel values to the range of [0, 1]
    train_dataset = vehicle_img(train_img / 255, train_typeID, transform = transforms.Compose([transforms.ToTensor()]))
    # Apply batch training with a batch size of 64
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

    # Normalize pixel values to the range of [0, 1]
    val_dataset = vehicle_img(val_img / 255, val_typeID, transform = transforms.Compose([transforms.ToTensor()]))
    val_loader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = True)

    # Normalize pixel values to the range of [0, 1]
    test_dataset = vehicle_img(test_img / 255, test_typeID, transform = transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = True)

    if args.model == 'fcn': # select fcn to train
        model = FCN(n_conv1=16, n_conv2=32, n_conv3=64, n_conv4=128, n_output=11, kernel_size=3)
    else: # select cnn to train
        model = CNN(n_conv1=16, n_conv2=32, n_lin1=8192, n_lin2=2000, n_output=11, kernel_size=3)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # Training base model
    u_train_loss_hist = []
    u_val_loss_hist = []
    for i in range(n_epoch):
        correct = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.float().to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target.long())
            train_loss += loss.item()
            _, pred = torch.max(F.softmax(output, 1), 1)
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        u_train_loss_hist.append(train_loss)
        print(args.model.upper() + ' Epoch', i + 1, ':')
        print('\nTrain loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
        val_loss = 0
        correct = 0
        for data, target in val_loader:
            data, target = data.float().cuda(), target.cuda()
            output = model(data)
            val_loss += F.cross_entropy(output, target.long(), reduction='sum').item()
            _, pred = torch.max(F.softmax(output, 1), 1)
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        val_loss /= len(val_loader.dataset)
        u_val_loss_hist.append(val_loss)
        print('Val loss: {:.4f}, Val Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

    # Rearrange train dataset for bi-directional training
    # One-hot encoding for target class labels in forward training flow
    type_one_hot = F.one_hot(train_typeID).float()
    # Add extra attribute to each training instance to make the task invertible
    type_extra_attr = np.concatenate((type_one_hot.numpy(), np.linspace(0, 1, len(type_one_hot), False).reshape(-1, 1)), 1)
    # Concatenate original target with modified target 
    # Original scalar class label and extra attribute would be the target of model in forward training flow 
    # One-hot vectors and extra attribute would be the input of model in reversed training flow
    type_target = np.concatenate((train_typeID.numpy().reshape(-1, 1), type_extra_attr), 1)
    type_target = torch.tensor(type_target, dtype = torch.float)

    # Normalize pixel value to the range of [0, 1]
    train_dataset = vehicle_img(train_img / 255, type_target, transform = transforms.Compose([transforms.ToTensor()]))
    # Apply batch training with a batch size of 64
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    
    if args.model == 'fcn': # select fcn to train
        # Model in forward training flow
        model_f = FCN(n_conv1=16, n_conv2=32, n_conv3=64, n_conv4=128, n_output=12, kernel_size=3)
        # Model in reversed training flow
        model_b = FCN_backward(n_conv1=16, n_conv2=32, n_conv3=64, n_conv4=128, n_output=12, kernel_size=3)
    else: # select cnn to train
        # Model in forward training flow
        model_f = CNN(n_conv1=16, n_conv2=32, n_lin1=8192, n_lin2=2000, n_output=12, kernel_size=3)
        # Model in reversed training flow
        model_b = CNN_backward(n_conv1=16, n_conv2=32, n_lin1=8192, n_lin2=2000, n_output=12, kernel_size=3)

    model_f.to(device)
    # Parameter optimizer for model in forward training flow
    optimizer_f = torch.optim.Adam(model_f.parameters(), lr = lr)
    model_b.to(device)
    # Parameter optimizer for model in reversed training flow
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr = lr)
    
    b_train_loss_hist = []
    b_val_loss_hist = []
    for i in range(n_epoch):
        correct = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.float().to(device), target.to(device)
            optimizer_f.zero_grad()
            optimizer_b.zero_grad()
            output = model_f(data)
            # Calculate cross entropy loss on class label prediction and MSE loss on extra attribute of forward model
            lf_ce = F.cross_entropy(output[:, :-1], target[:, 0].long())
            lf_mse = F.mse_loss(output[:, -1], target[:, -1])
            train_loss += lf_ce.item()
            _, pred = torch.max(F.softmax(output[:, :-1], 1), 1) # output[:, :-1].data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target[:, 0].data.view_as(pred)).long().cpu().sum()
            # Perform backprop
            lf_ce.backward(retain_graph = True)
            lf_mse.backward()
            # Update params of forward model
            optimizer_f.step()
            # Sync the params in reversed model with updated params in forward model
            sync_param_f2b(model_f, model_b, args.model)
            # Calculate MSE loss of the reversed model according to the input of forward model
            lb = F.mse_loss(model_b(target[:, 1:]), data)
            lb.backward()
            # Update params of reversed model
            optimizer_b.step()
            # Sync the params in forward model with updated params in reversed model
            sync_param_b2f(model_f, model_b, args.model)

        train_loss /= len(train_loader)
        b_train_loss_hist.append(train_loss)
        print('BD' + args.model.upper() + ' Epoch', i + 1, ':')
        print('\nTrain loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
                
        val_loss = 0
        correct = 0
        for data, target in val_loader:
            data, target = data.float().to(device), target.to(device)
            # Ignore the results of predicted extra attribute in validation set
            output = model_f(data)[:, :-1]
            val_loss += F.cross_entropy(output, target.long(), reduction = 'sum').item()
            _, pred = torch.max(F.softmax(output, 1), 1)
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        val_loss /= len(val_loader.dataset)
        b_val_loss_hist.append(val_loss)
        print('Val loss: {:.4f}, Val Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

    # Model testing
    # Calculate test loss and accuracy for base model
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.float().to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target.long(), reduction='sum').item()
        _, pred = torch.max(F.softmax(output, 1), 1)
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n' + args.model.upper() + ' Test loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Calculate test loss and accuracy for bi-directional model
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.float().to(device), target.long().to(device)
        # Ignore the results of extra attribute in test set
        output = model_f(data)[:, :-1]
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        _, pred = torch.max(F.softmax(output, 1), 1)
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nBD' + args.model.upper() + ' Test loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Plot train and validation loss curves
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(range(n_epoch), u_train_loss_hist, label = 'Train')
    ax1.plot(range(n_epoch), u_val_loss_hist, label = 'Val')
    ax1.legend()
    ax1.set_title(args.model.upper() + ' Loss', fontsize = 16)
    ax2 = fig.add_subplot(122)
    ax2.plot(range(n_epoch), b_train_loss_hist, label = 'Train')
    ax2.plot(range(n_epoch), b_val_loss_hist, label = 'Val')
    ax2.legend()
    ax2.set_title('BD' + args.model.upper() + ' Loss', fontsize = 16)
    ax2.legend()
    plt.show()
