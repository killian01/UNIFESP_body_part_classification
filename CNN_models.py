import torch
import torch.nn as nn
from torchvision import models


class Resnet50Multilabels(nn.Module):
    def __init__(self):
        super(Resnet50Multilabels, self).__init__()
        
        # get the pretrained VGG16 network
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 22)
        
        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
        #get avgpool
        self.avg_pool = self.resnet.avgpool
        
        # get the classifier of the vgg16
        self.fc = self.resnet.fc
        
        # placeholder for the gradients
        self.gradients = None

        #Sigmoid
        self.sig = nn.Sigmoid()

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        # apply the remaining pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), 1*1*2048)
        x = self.fc(x)
        x = self.sig(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    def get_feature_vector(self, x):
        x = self.features_conv(x)
        x = self.avg_pool(x)       
        x = x.view(x.size(0), 1*1*2048)
        return x
    
    # method for the activation extraction
    def get_activations(self, x):
        return self.features_conv(x)

