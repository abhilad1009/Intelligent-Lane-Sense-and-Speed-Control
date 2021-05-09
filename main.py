from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from absl import app, flags
from absl.flags import FLAGS

#Local imports
from dataloader import load_data
from load_mobilenet import load_mobilenetv2
from model import prep_model
from train import train_model
from test import test_model
from confusion_matrix import calculate_confusion_matrix


flags.DEFINE_string('root',"D:\\Github\\cloned repo\\Intelligent-Lane-Sense-and-Speed-Control",'Root folder path')
flags.DEFINE_string('dataroot',"D:\College\Intelligent Lane sense\code\data",'path to root folder of dataset folder')
flags.DEFINE_integer('batch_size',128,'Input batch size')
flags.DEFINE_integer('image_size',224,'Image size input (default 224*224)')
flags.DEFINE_boolean('save_model',True,'True if you want to train the model and save it')
flags.DEFINE_integer('num_epoch',5,'Number of Epoches')
flags.DEFINE_boolean('train',False,'True if you want to train the model')
flags.DEFINE_boolean('test',False,'True if you want to test the model')

def main(_argv):
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    #Hyperparameter
    # Root directory for dataset
    dataroot =  FLAGS.dataroot

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = FLAGS.batch_size

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = FLAGS.image_size

    #number of epoches
    num_epoch=FLAGS.num_epoch

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    #Load dataset
    dataloader, device = load_data(dataroot , image_size , batch_size, workers, ngpu) 

    
    #load mobilenet-v3
    model= load_mobilenetv2()

    #print model layers
    # para=model.state_dict()
    # print(para.keys())

    #Freezing all layers except the last layer. Also, filter the layers while updating the weights.
    for name, param in model.named_parameters():
        if param.requires_grad and not 'classifier.1.bias' in name:
            param.requires_grad = False
            #print(name,param.requires_grad)
    
    model.classifier=torch.nn.Identity()


    
    # para=model.state_dict()
    # print(para.keys())
    
    
    #Add softmax layer
    model=prep_model(model)

    print(model)

    if(FLAGS.train == True):
        #Start training

        model=train_model(model,dataloader, num_epoch,device)

        if(FLAGS.save_model):

            print("Saving models...")

            try:
                torch.save(model.state_dict(), FLAGS.root + "\\trained_model\\model.h5")
        
                print("Model is saved...")
            except Exception as e:
                print("Error while saving model: ",e)
                exit()

    if(FLAGS.test == True):
        
        print("Loading models weights...")
        try:
            model.load_state_dict(torch.load(FLAGS.root + "\\trained_model\\model.h5"))

            print("Weights Loaded...")
        except Exception as e:
            print("Error while loading the model:", e)
            exit()
            
        calculate_confusion_matrix(model,dataloader,device)
        #test_model(model,dataloader,device)


    
            
            



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass