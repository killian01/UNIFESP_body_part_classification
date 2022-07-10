import torch
import torch.nn as nn
from torchvision import models


class VGG_binary(nn.Module):
    def __init__(self):
        super(VGG_binary, self).__init__()
        
        # get the pretrained VGG16 network
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, 2)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:30]
        
        # get the max pool of the features stem
        #self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        #get avgpool
        self.avg_pool = self.vgg.avgpool
        
        # get the classifier of the vgg16
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
        
        
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = self.avg_pool(x)
        
        x = x.view(x.size(0), 7*7*512)
        #x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    def get_feature_vector(self, x):
        x = self.features_conv(x)
        
        # register the hook
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
            
        # apply the remaining pooling
        x = self.max_pool(x)
        x = self.avg_pool(x)
        
        x = x.view(x.size(0), 7*7*512)
        #x = self.first_linear_layer(x)
        x = self.vgg.classifier[:2](x)
        
        return x
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        # get the pretrained VGG16 network
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, 4)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:30]
        
        # get the max pool of the features stem
        #self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        #get avgpool
        self.avg_pool = self.vgg.avgpool
        
        # get the classifier of the vgg16
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = self.avg_pool(x)
        
        x = x.view(x.size(0), 7*7*512)
        #x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
    
class VGG_multi_labels(nn.Module):
    def __init__(self):
        super(VGG_multi_labels, self).__init__()
        
        # get the pretrained VGG16 network
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, 2)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:30]
        
        # get the max pool of the features stem
        #self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        #get avgpool
        self.avg_pool = self.vgg.avgpool
        
        # get the classifier of the vgg16
        self.classifier = self.vgg.classifier
        
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
        x = self.max_pool(x)
        x = self.avg_pool(x)
        
        x = x.view(x.size(0), 7*7*512)
        #x = x.view((1, -1))
        x = self.classifier(x)
        
        x = self.sig(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    

class Linear_model(nn.Module):
    def __init__(self):
        super(Linear_model, self).__init__()
        
#         # get the pretrained VGG16 network
#         self.vgg = models.vgg16(pretrained=True)
#         self.vgg.classifier[6] = nn.Linear(4096, 2)
        
#         self.classifier = self.vgg.classifier
        

#         self.linear_layer = nn.Sequential(
#              nn.Linear(69632, 25088, bias=True),
#              nn.ReLU(inplace=True),
#              nn.Dropout(p=0.5, inplace=False),
#         )

        self.linear_layer = nn.Sequential(
             nn.Linear(34816, 4096),
             nn.ReLU(),
             nn.Dropout(),
             nn.Linear(4096, 1024),
             nn.ReLU(),
             nn.Dropout(),
             nn.Linear(1024, 2),
        )
        
    def forward(self, x):
        x = self.linear_layer(x)
        return x



class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        
        # get the pretrained VGG16 network
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 2)
        
        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
        #get avgpool
        self.avg_pool = self.resnet.avgpool
        
        # get the classifier of the vgg16
        self.fc = self.resnet.fc
        
        # placeholder for the gradients
        self.gradients = None
        
        
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
        #x = x.view((1, -1))
        #x = torch.flatten(x)
        x = self.fc(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    def get_feature_vector(self, x):
        x = self.features_conv(x)
        x = self.avg_pool(x)       
        x = x.view(x.size(0), 1*1*2048)
        return x
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
    
    
class Linear_Resnet_features(nn.Module):
    def __init__(self):
        super(Linear_Resnet_features, self).__init__()
        self.linear_layer = nn.Sequential(
             nn.Linear(4096, 2048),
             nn.ReLU(),
             nn.Dropout(),
             nn.Linear(2048, 4),
        )
        
    def forward(self, x):
        x = self.linear_layer(x)
        return x