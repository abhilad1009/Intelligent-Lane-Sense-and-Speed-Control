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
from torchvision.utils import make_grid
#Local imports
from dataloader import load_data
from sklearn.metrics import confusion_matrix, accuracy_score

flags.DEFINE_string('root',"D:\\Github\\cloned repo\\Intelligent-Lane-Sense-and-Speed-Control",'Root folder path')
flags.DEFINE_string('dataroot',"D:\\College\\Intelligent Lane sense\\dataset\\trainedge-20210528T090645Z-001\\trainedge",'path to root folder of dataset folder')
flags.DEFINE_integer('batch_size',128,'Input batch size')
flags.DEFINE_integer('image_size',224,'Image size input (default 224*224)')
flags.DEFINE_boolean('save_model',True,'True if you want to train the model and save it')
flags.DEFINE_integer('num_epoch',5,'Number of Epoches')
flags.DEFINE_boolean('train',False,'True if you want to train the model')
flags.DEFINE_boolean('test',False,'True if you want to test the model')


def show_tensor_images(image_tensor, num_images=25, size=(3, 64, 64)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    
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
    
    #torch.set_printoptions(profile="full")
    count_1=0
    count_2=0
    total_images=0
    nb_classes = 3

    predlist=[]
    lbllist=[]
    # Initialize the prediction and label lists(tensors)
    # predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    # lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    for image, labels in dataloader:

        print(labels)
        
        plt.show()
        #print(image[0][0].size())
        
        pred_corr=0

        for k in range(batch_size):
            
            for i in range(224):
                for j in range(224):
                    if image[k][0][i][j] > 0:
                        if j>112:
                            count_2=count_2+1
                        else:
                            count_1=count_1+1
            if abs(count_1-count_2) < 100:
                pred=2
                
            else:
                if count_1>count_2:
                    pred=1    
                else:
                    pred=0

            # predlist=torch.cat([predlist,pred])
            # lbllist=torch.cat([lbllist,labels[k].item()])
            predlist.append(pred)
            lbllist.append(labels[k].item())

            #print(f"Image {k}: Left half ={count_1}, Right half ={count_2}, Prediction class: {pred} Actual Class ={labels[k]}")
            #print("Image ",k, ":",count_1,count_2,labels[k])
            count_1=0
            count_2=0
            
        print(predlist,lbllist)
        #show_tensor_images(image)
        break


    conf_mat=confusion_matrix(lbllist, predlist)
    print(conf_mat)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print("Class-wise accuracy=",class_accuracy)

    accuracy=accuracy_score(lbllist, predlist)
    print("Accuracy=",accuracy)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass