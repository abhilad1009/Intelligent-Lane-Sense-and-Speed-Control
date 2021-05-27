import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def train_model(model, dataloader, num_epochs ,device):

    criterion=nn.NLLLoss()
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)



    print("Starting to train..")
    
    for epoch in range(num_epochs):
        count=0
        # For each batch in the dataloader
        for images, labels in dataloader:

            model.zero_grad()

            output= model(images)

            #labels=one_hot = torch.nn.functional.one_hot(labels, 3) 
            # print("Output=",output.size())
            # print("Labels",labels.size())

            #print(output)
            loss=criterion(torch.log(output),labels)

            loss.backward()

            optimizer.step()

            #if count % 50 == 0:
            print("Epoches:[",epoch,"/",num_epochs,"]", "\tBatch:[",count,"/", len(dataloader),"]","\tCategorial cross entropy loss=",loss.item())
        

            count=count+1
    return model

                
                
                

