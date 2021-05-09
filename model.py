import torch
from torch import nn

class prep_model(nn.Module):
    def __init__(self, model):
        super(prep_model, self).__init__()

        self.first_layer=nn.Sequential(
            nn.Conv2d(1,3,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )

        self.pretrained=model

        self.last_layer=nn.Sequential(
            nn.Linear(1280,3),
            
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        x=self.first_layer(x)
        x=self.pretrained(x)
        x=self.last_layer(x)

        return x