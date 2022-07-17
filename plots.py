# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:39:30 2022

@author: lasse
"""

""" 
    This script contains plot functions
    used for evaluating the image recognition algorithm

"""

import matplotlib.pyplot as plt
import numpy as np
import math
import torchvision

def loss_plot(epochs, loss):
    fig, ax = plt.subplots(1, dpi=200)
    ax.plot(np.linspace(1, epochs, epochs), loss)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch No.')
    plt.show()

def accuracy_plot(epochs, accuracy):
    fig, ax = plt.subplots(1, dpi=200)
    ax.plot(np.linspace(1, epochs, epochs), accuracy)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Epoch No.')
    plt.show()

def perfomance_plot(epochs, train_accuracy, test_accuracy, train_losses, test_losses):
    from matplotlib.ticker import MultipleLocator
    fig, ax = plt.subplots(1, 2, figsize=(12,5), dpi=200)
    ax[0].plot(np.linspace(1, epochs, epochs), train_accuracy, label='Training')
    ax[0].plot(np.linspace(1, epochs, epochs), test_accuracy, label='Validation')
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].set_xlabel('Epoch No.')
    ax[0].legend(loc='upper left')
    ax[0].grid(which='major', axis='y')
    ax[0].grid(which='minor', ls='--', lw=0.5)
    ax[0].set_ylim(0, 100)
    ax[0].yaxis.set_minor_locator(MultipleLocator(10))
    ax[0].xaxis.set_minor_locator(MultipleLocator(10))
    
    
    ax[1].plot(np.linspace(1, epochs, epochs), train_losses, label='Training')
    ax[1].plot(np.linspace(1, epochs, epochs), test_losses, label='Validation')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch No.')
    ax[1].legend(loc='upper right')
    ax[1].grid(which='major', axis='y')
    ax[1].grid(which='minor', ls='--', lw=0.5)
    ax[1].xaxis.set_minor_locator(MultipleLocator(10))
    plt.show()

def model_testrun(dataloader, test_size, classes):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    # show images
    fig, ax = plt.subplots(2, 1, dpi=200)
    ax[1].imshow(torchvision.utils.make_grid(images).permute(1, 2, 0), aspect='auto')
    # print labels
    
    y_increment = math.ceil(test_size/8)
    y = y_increment/(y_increment+1)
    j=0
    fig.suptitle('Evaluation of NN perfomance with test set')
    ax[1].set_title('Images analyzed', fontsize=8)
    ax[0].set_title('NN Labeling of images below', fontsize=8)
    for i in range(test_size):
        if i % 8 == 0 and i != 0:
            j=0
            y -= 1/(y_increment+1)
        ax[0].text(1/16 + 1/8*j, y, classes[labels[i]], fontsize=6, ha='center', va='center')
        j+=1
        
    for axis in ax.reshape(-1):
        axis.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False) # labels along the bottom edge are off
    plt.show()

def confusion_matrix(model, test_dl, classes, is_normalized=True, cuda_compatible = False):
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    import torch

    y_pred = []
    y_true = []
    # iterate over test data
    for data in test_dl:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            _, labelss = data
            output = model(inputs)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth
    
    counts = [list(labelss).count(x) for x in range(0, 6)]
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    if is_normalized:
        df_cm = pd.DataFrame([(cf_matrix[:,x]/counts[x]) * 100 for x in range(0,6)], index = [i for i in classes],
                             columns = [i for i in classes])
    else:
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                             columns = [i for i in classes])
    plt.figure(dpi=200, figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


    def bin_plot(dataset):
        n_classes = len(dataset.classes)
        classes = dataset.classes
        counts = [dict(Counter(dataset.targets))[x] for x in np.arange(0, n_classes, 1)]
        plt.rcParams['figure.figsize'] = (8, 5)
        plt.rcParams['figure.dpi'] = 200
        plt.bar(list(dict(Counter(dataset.targets))), counts)
        plt.xticks(np.arange(0, n_classes, 1), classes)
        plt.xlabel('Class', fontsize=16)
        plt.ylabel('Bin Size', fontsize=16)
        plt.grid('on', axis='y')
        plt.title(f'Total dataset size: {sum(counts)}')
        plt.savefig('Datapool.png', dpi=200)
        plt.show()