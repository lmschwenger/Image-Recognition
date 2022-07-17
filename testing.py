
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:41:23 2022

@author: lasse
"""
import torch
import torch.nn as nn
from CNN import AverageMeter, accuracy
def model_testing(model, test_dataloader, cuda_compatible=False):
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss=[]; test_accuracy=[]
    
    with torch.no_grad():
        for data in test_dataloader:
            data_input, target = data
            if cuda_compatible:
                data_input, target = data_input.cuda(), target.cuda()
            # print(data_input.size())
            # print(target.size())
            output = model(data_input) # data_input.view(-1, data_input.size()[-2]*data_input.size()[-1])
            
            for idx, i in enumerate(output):
                if torch.argmax(i) == target[idx]:
                    correct += 1
                loss = criterion(output, target)
                test_loss.append(loss.item())
                acc1,_=accuracy(output, target, topk=(1, 5))
                test_accuracy.append(acc1[0])
                total += 1
                
                
    return sum(test_loss)/total, (correct/total)*100