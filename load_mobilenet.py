import torch

def load_mobilenetv2():
    
    #If you don't have mobilenet-v2 in your torchvision then this will download it for you otherwise use the cached one
    # Make sure you have the torchvision that has mobilenet-v2  module. This repository has used torchvision version 0.9
    model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    
    model.eval()
    return model