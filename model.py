import torch
import torch.nn as nn

class CNN(nn.Module):
    """ Convolutional neural network. """
    def __init__(self, n_conv1, n_conv2, n_lin1, n_lin2, n_output, kernel_size):
        super(CNN, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, n_conv1, kernel_size=kernel_size, padding=2)
        # Pooling layer for conv layer 1
        self.conv1_pool = nn.Conv2d(n_conv1, n_conv1, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(n_conv1, n_conv2, kernel_size=kernel_size, padding=2)
        # Pooling layer for conv layer 2
        self.conv2_pool = nn.Conv2d(n_conv2, n_conv2, kernel_size=4, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(n_lin1, n_lin2)
        self.fc2 = nn.Linear(n_lin2, n_output)
        self.n_lin1 = n_lin1 # For feature flatten purpose
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1_pool(self.relu(self.conv1(x)))
        x = self.conv2_pool(self.relu(self.conv2(x)))
        # Flatten extracted features from conv layers
        x = x.view(-1, self.n_lin1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_backward(nn.Module):
    """ A reversed convolutional neural network for bi-directional training. """
    def __init__(self, n_conv1, n_conv2, n_lin1, n_lin2, n_output, kernel_size):
        super(CNN_backward, self).__init__()
        # Reverse of conv pooling layer 2
        self.conv2_t = nn.ConvTranspose2d(n_conv2, n_conv1, kernel_size=kernel_size, padding=2)
        self.conv2_pool_t = nn.ConvTranspose2d(n_conv2, n_conv2, kernel_size=4, stride=2)
        self.conv1_t = nn.ConvTranspose2d(n_conv1, 1, kernel_size=kernel_size, padding=2)
        # Reverse of conv pooling layer 1
        self.conv1_pool_t = nn.ConvTranspose2d(n_conv1, n_conv1, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(n_lin1, n_lin2)
        self.fc2 = nn.Linear(n_lin2, n_output)
        self.n_conv2 = n_conv2 # For changing dimensionality purpose
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Training in the reverse flow of conventional CNN
        x = self.relu(torch.matmul(x, self.fc2.weight) + self.fc1.bias)
        x = torch.matmul(x, self.fc1.weight).view(-1, self.n_conv2, 16, 16) + self.conv2_pool_bias.view(1, self.n_conv2, 1, 1)
        x = self.relu(self.conv2_pool_t(x))
        x = self.conv2_t(x)
        x = self.relu(self.conv1_pool_t(x))
        x = self.conv1_t(x)
        return x

    def set_conv2_pool_bias(self, bias):
        # Set the bias between conv pooling layer 2 and linear layer 1
        self.conv2_pool_bias = nn.Parameter(bias)


class FCN(nn.Module):
    """ Fully convolutional network. """
    def __init__(self, n_conv1, n_conv2, n_conv3, n_conv4, n_output, kernel_size):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(1, n_conv1, kernel_size=kernel_size, stride=1, padding=2)
        # Pooling layer for conv layer 1
        self.conv1_pool = nn.Conv2d(n_conv1, n_conv1, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(n_conv1, n_conv2, kernel_size=kernel_size, stride=1, padding=2)
        # Pooling layer for conv layer 2
        self.conv2_pool = nn.Conv2d(n_conv2, n_conv2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(n_conv2, n_conv3, kernel_size=kernel_size, stride=1, padding=2)
        # Pooling layer for conv layer 3
        self.conv3_pool = nn.Conv2d(n_conv3, n_conv3, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(n_conv3, n_conv4, kernel_size=kernel_size, stride=1, padding=2)
        # Pooling layer for conv layer 4
        self.conv4_pool = nn.Conv2d(n_conv4, n_conv4, kernel_size=4, stride=2)
        # Output convolution layer
        # Convert output feature maps to size of 1 * 1
        self.conv5 = nn.Conv2d(n_conv4, n_output, kernel_size=4)
        self.n_output = n_output # For feature flatten purpose
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1_pool(self.relu(self.conv1(x)))
        x = self.conv2_pool(self.relu(self.conv2(x)))
        x = self.conv3_pool(self.relu(self.conv3(x)))
        x = self.conv4_pool(self.relu(self.conv4(x)))
        x = self.conv5(x)
        return x.view(-1, self.n_output) # Flatten feature maps to vectors


class FCN_backward(nn.Module):
    """ A reversed fully convolutional network for bi-directional training. """
    def __init__(self, n_conv1, n_conv2, n_conv3, n_conv4, n_output, kernel_size):
        super(FCN_backward, self).__init__()
        self.conv5_t = nn.ConvTranspose2d(n_output, n_conv4, kernel_size=4)
        # Reverse of conv pooling layer 4
        self.conv4_pool_t = nn.ConvTranspose2d(n_conv4, n_conv4, kernel_size=4, stride=2)
        self.conv4_t = nn.ConvTranspose2d(n_conv4, n_conv3, kernel_size=kernel_size, padding=2)
        # Reverse of conv pooling layer 3
        self.conv3_pool_t = nn.ConvTranspose2d(n_conv3, n_conv3, kernel_size=4, stride=2)
        self.conv3_t = nn.ConvTranspose2d(n_conv3, n_conv2, kernel_size=kernel_size, padding=2)
        # Reverse of conv pooling layer 2
        self.conv2_pool_t = nn.ConvTranspose2d(n_conv2, n_conv2, kernel_size=4, stride=2)
        self.conv2_t = nn.ConvTranspose2d(n_conv2, n_conv1, kernel_size=kernel_size, padding=2)
        # Reverse of conv pooling layer 1
        self.conv1_pool_t = nn.ConvTranspose2d(n_conv1, n_conv1, kernel_size=4, stride=2)
        self.conv1_t = nn.ConvTranspose2d(n_conv1, 1, kernel_size=kernel_size, padding=2)
        self.n_output = n_output # For changing dimensionality purpose
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Change input vectors to 1 * 1 feature maps
        x = x.view(-1, self.n_output, 1, 1)
        x = self.conv5_t(x)
        x = self.relu(self.conv4_pool_t(x))
        x = self.conv4_t(x)
        x = self.relu(self.conv3_pool_t(x))
        x = self.conv3_t(x)
        x = self.relu(self.conv2_pool_t(x))
        x = self.conv2_t(x)
        x = self.relu(self.conv1_pool_t(x))
        x = self.conv1_t(x)
        return x

def sync_param_f2b(model_f, model_b, model_type):
    """ Parameter synchronization from forward model to reversed model. """
    # Weights between corresponding layers are shared
    # Biases of a reversed layer is synchronized with that of the previous 
    # layer of its corresponding layer in forward model
    model_b.conv1_t.weight = model_f.conv1.weight
    model_b.conv1_pool_t.weight = model_f.conv1_pool.weight
    model_b.conv1_pool_t.bias = model_f.conv1.bias
    model_b.conv2_t.weight = model_f.conv2.weight
    model_b.conv2_t.bias = model_f.conv1_pool.bias
    model_b.conv2_pool_t.weight = model_f.conv2_pool.weight
    model_b.conv2_pool_t.bias = model_f.conv2.bias
    if model_type == 'fcn': # sync for FCN model
        model_b.conv3_t.weight = model_f.conv3.weight
        model_b.conv3_t.bias = model_f.conv2_pool.bias
        model_b.conv3_pool_t.weight = model_f.conv3_pool.weight
        model_b.conv3_pool_t.bias = model_f.conv3.bias
        model_b.conv4_t.weight = model_f.conv4.weight
        model_b.conv4_t.bias = model_f.conv3_pool.bias
        model_b.conv4_pool_t.weight = model_f.conv4_pool.weight
        model_b.conv4_pool_t.bias = model_f.conv4.bias
        model_b.conv5_t.weight = model_f.conv5.weight
        model_b.conv5_t.bias = model_f.conv4_pool.bias
    else: # sync for CNN model
        model_b.set_conv2_pool_bias(model_f.conv2_pool.bias)
        model_b.fc1.weight = model_f.fc1.weight
        model_b.fc1.bias = model_f.fc1.bias
        model_b.fc2.weight = model_f.fc2.weight

def sync_param_b2f(model_f, model_b, model_type):
    """ Parameter synchronization from reversed CNN model to forward CNN model. """
    # Weights between corresponding layers are shared
    # Biases of a forward layer is synchronized with that of the next 
    # layer of its corresponding layer in reversed model
    model_f.conv1.weight = model_b.conv1_t.weight
    model_f.conv1.bias = model_b.conv1_pool_t.bias
    model_f.conv1_pool.weight = model_b.conv1_pool_t.weight
    model_f.conv1_pool.bias = model_b.conv2_t.bias
    model_f.conv2.weight = model_b.conv2_t.weight
    model_f.conv2.bias = model_b.conv2_pool_t.bias
    model_f.conv2_pool.weight = model_b.conv2_pool_t.weight
    if model_type == 'fcn': # sync for FCN models
        model_f.conv2_pool.bias = model_b.conv3_t.bias
        model_f.conv3.weight = model_b.conv3_t.weight
        model_f.conv3.bias = model_b.conv3_pool_t.bias
        model_f.conv3_pool.weight = model_b.conv3_pool_t.weight
        model_f.conv3_pool.bias = model_b.conv4_t.bias
        model_f.conv4.weight = model_b.conv4_t.weight
        model_f.conv4.bias = model_b.conv4_pool_t.bias
        model_f.conv4_pool.weight = model_b.conv4_pool_t.weight
        model_f.conv4_pool.bias = model_b.conv5_t.bias
        model_f.conv5.weight = model_b.conv5_t.weight
    else: # sync for CNN models
        model_f.conv2_pool.bias = model_b.conv2_pool_bias
        model_f.fc1.weight = model_b.fc1.weight
        model_f.fc1.bias = model_b.fc1.bias
        model_f.fc2.weight = model_b.fc2.weight

# def fcn_sync_weight_f2b(model_f, model_b):
#     """ Parameter synchronization from forward FCN model to reversed FCN model. """
#     model_b.conv1_t.weight = model_f.conv1.weight
#     model_b.conv1_pool_t.weight = model_f.conv1_pool.weight
#     model_b.conv1_pool_t.bias = model_f.conv1.bias
#     model_b.conv2_t.weight = model_f.conv2.weight
#     model_b.conv2_t.bias = model_f.conv1_pool.bias
#     model_b.conv2_pool_t.weight = model_f.conv2_pool.weight
#     model_b.conv2_pool_t.bias = model_f.conv2.bias
#     model_b.conv3_t.weight = model_f.conv3.weight
#     model_b.conv3_t.bias = model_f.conv2_pool.bias
#     model_b.conv3_pool_t.weight = model_f.conv3_pool.weight
#     model_b.conv3_pool_t.bias = model_f.conv3.bias
#     model_b.conv4_t.weight = model_f.conv4.weight
#     model_b.conv4_t.bias = model_f.conv3_pool.bias
#     model_b.conv4_pool_t.weight = model_f.conv4_pool.weight
#     model_b.conv4_pool_t.bias = model_f.conv4.bias
#     model_b.conv5_t.weight = model_f.conv5.weight
#     model_b.conv5_t.bias = model_f.conv4_pool.bias

# def fcn_sync_weight_b2f(model_f, model_b):
#     model_f.conv1.weight = model_b.conv1_t.weight
#     model_f.conv1.bias = model_b.conv1_pool_t.bias
#     model_f.conv1_pool.weight = model_b.conv1_pool_t.weight
#     model_f.conv1_pool.bias = model_b.conv2_t.bias
#     model_f.conv2.weight = model_b.conv2_t.weight
#     model_f.conv2.bias = model_b.conv2_pool_t.bias
#     model_f.conv2_pool.weight = model_b.conv2_pool_t.weight
#     model_f.conv2_pool.bias = model_b.conv3_t.bias
#     model_f.conv3.weight = model_b.conv3_t.weight
#     model_f.conv3.bias = model_b.conv3_pool_t.bias
#     model_f.conv3_pool.weight = model_b.conv3_pool_t.weight
#     model_f.conv3_pool.bias = model_b.conv4_t.bias
#     model_f.conv4.weight = model_b.conv4_t.weight
#     model_f.conv4.bias = model_b.conv4_pool_t.bias
#     model_f.conv4_pool.weight = model_b.conv4_pool_t.weight
#     model_f.conv4_pool.bias = model_b.conv5_t.bias
#     model_f.conv5.weight = model_b.conv5_t.weight


