import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def test_model(model,dataloader,device):

    total_output=[]
    total_label=[]

    correct_pred=0
    total_images=128*len(dataloader)

    print("Start testing....")
    print("Total Number of test images: ",total_images)

    count=0
    for images, labels in dataloader:
        with torch.no_grad():
            print("Batch=",count,"Number of images=",len(images))
            output = model(images)
            
            for i in range(len(images)):
                if torch.argmax(output[i],dim=0).item() == labels[i].item():
                    correct_pred = correct_pred + 1

        count=count+1

    
    accuracy=correct_pred/total_images*100

    print("Accuracy: ",accuracy,"%")
    

