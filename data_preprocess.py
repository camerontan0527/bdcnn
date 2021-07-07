import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Note that in this file, we assume the pixel values of all images are in the range of [0, 255]

def load_img(file_list, dir_path):
    """ Load all images under a directory to a numpy array. """
    data = []
    for file in file_list:
        img = plt.imread(dir_path + file)
        # Convert RGB image to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Resize image to desired size
        img = cv2.resize(img, (64, 64))
        # Store processed image to list
        data.append(img)
    return np.array(data)

path = './data/'
# Load all file names under a directory
train_files = os.listdir(path + 'train')
test_files = os.listdir(path + 'test')
val_files = os.listdir(path + 'val')

# Load train, test, validation images to numpy arrays
train_data = load_img(train_files, path + 'train/')
test_data = load_img(test_files, path + 'test/')
val_data = load_img(val_files, path + 'val/')

# Save preprocessed data to npy files
np.save(path + 'train_data.npy', train_data)
np.save(path + 'test_data.npy', test_data)
np.save(path + 'val_data.npy', val_data)
