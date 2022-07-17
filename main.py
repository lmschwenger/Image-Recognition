# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:58:56 2022

@author: lasse
"""


""" Setting directory to folder of this file """
import os

current_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_folder)

#Imports 
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision import transforms, datasets
from skimage import io, transform
import pandas as pd
import plots
import numpy as np
from collections import Counter
#%% Loading with ImageFolder extension
from CNN import RandomRotation, RandomShift
size=128 #Setting resize size of pictures. PyTorch cannot evaluate pictures with varying dimensions

my_transforms = transforms.Compose([transforms.Resize(128),
                                  transforms.CenterCrop(size),
                                  transforms.ToTensor(),                                  
                                  transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])])

dataset = datasets.ImageFolder(current_folder + '/Test Images/', transform=my_transforms)
# Specifying training and test sizes and batch sizes for training
training_size = int(np.floor(len(dataset) * 0.8))
test_size = len(dataset) - training_size
my_batch_size = 780

train, test = torch.utils.data.random_split(dataset, [training_size, test_size])
train_dl = torch.utils.data.DataLoader(dataset=train, batch_size=my_batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(dataset=test, batch_size=my_batch_size, shuffle=True)

#%% Visualizing class sizes
#plots.bin_plot(dataset)

#%%Training the model
from training import model_training
from CNN import Net
from torch.optim import lr_scheduler

model = Net(size, possible_outcomes=6)
criterion = nn.CrossEntropyLoss()
EPOCHS = 1000    #Training the model
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
cuda_compatible=False

# Checking if CUDA is available. If true it alters the process slightly to be CUDA-compatible.
if torch.cuda.is_available():
    print("Using CUDA")
    torch.device("cuda")
    model = model.cuda()
    criterion = criterion.cuda()
    cuda_compatible=True

model_training(model, train_dl, test_dl, EPOCHS, criterion, optimizer, perfomance_plot=False, cuda_compatible=cuda_compatible)
# Example data
if input("Show example data? (y/n)") == "y":
    dataiter = iter(test_dl)
    images, labels = dataiter.next()
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(12, 3))
    ax.imshow(torchvision.utils.make_grid(images[:5]).permute(1, 2, 0), aspect='auto')

#%% Loading pretrained model
from training import model_training
from CNN import Net
from torch.optim import lr_scheduler


# Confusion Matrix
from plots import confusion_matrix


if input("Show Confusion matrix ? (y/n)") == "y":
    if input("In percent or count? (pct/nr)") == 'pct':
        confusion_matrix(model, test_dl, classes=dataset.classes, is_normalized=True, cuda_compatible=cuda_compatible)
    else:
        confusion_matrix(model, test_dl, classes=dataset.classes, is_normalized=False, cuda_compatible=cuda_compatible)

# Saving the model
if input("Do you want to save and overwrite the model? (y/n)") == "y":
    torch.save(model, 'model_weights.pth')
    
    

