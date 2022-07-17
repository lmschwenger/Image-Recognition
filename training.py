
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:31:14 2022

@author: lasse
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import plots
import os
from CNN import AverageMeter, ProgressMeter, accuracy
from testing import model_testing
import torch.nn as nn
import time

""" Setting directory to folder of this file """
current_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_folder)

def model_training(model, training_dataloader, testing_dataloader, num_epochs, criterion, optimizer, loss_plot=0, accuracy_plot=0, perfomance_plot=1, cuda_compatible=False):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    correct=0; total=0
    
    train_loss=[]; train_accuracy=[]; test_accuracies=[]; test_losses=[]
    for epoch in range(num_epochs):
        batch_time = AverageMeter('Time', ':3.3f')
        data_time = AverageMeter('Data', ':3.3f')
        losses = AverageMeter('Loss', ':.2e')
        top1 = AverageMeter('Acc@1', ':2.2f')
        acc_test = AverageMeter('acc_test', ':2.2f')
        
        progress = ProgressMeter(num_epochs,
            [batch_time, data_time, losses, top1, acc_test],
            prefix="Epoch: [{}]".format(epoch+1))        
        
        epoch_loss=[]
        test_loss, test_accuracy = model_testing(model, testing_dataloader, cuda_compatible)
        
        test_accuracies.append(test_accuracy)
        
        test_losses.append(test_loss)
        end = time.time()
        for data in training_dataloader:
            
            X, labels = data
            X, labels = X.cuda(), labels.cuda()
            if torch.cuda.is_available():
                X = X.cuda()
                labels = labels.cuda()
            model.zero_grad()
            output = model(X) # X.view(-1, X.size()[-2]*X.size()[-1]).type(torch.float32)
            
            for idx, i in enumerate(output):
            
                if torch.argmax(i) == labels[idx]:
                
                    correct += 1
                
                total += 1
            if criterion == 'nll_loss':
                loss = F.nll_loss(output, labels)
            else:
                loss = criterion(output, labels)
            losses.update(loss.item(), X.size(0))
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], X.size(0))
            acc_test.update(test_accuracy, len(testing_dataloader))            
            
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
                    # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
        progress.display(epoch+1)
        
        train_loss.append(loss.item())
        train_accuracy.append(acc1)

    if loss_plot:
        plots.accuracy_plot(num_epochs, train_accuracy)
    if accuracy_plot:
        plots.loss_plot(num_epochs, train_loss)
    
    if perfomance_plot:
        print(train_accuracy)
        if cuda_compatible:
            train_accuracy = torch.tensor(train_accuracy, device='cpu')
            test_accuracy = torch.tensor(test_accuracy, device='cpu')
            train_loss = torch.tensor(train_loss, device='cpu')
            test_losses = torch.tensor(test_losses, device='cpu')
        plots.perfomance_plot(num_epochs, 
                              train_accuracy,
                              test_accuracies, 
                              train_loss, 
                              test_losses)
    