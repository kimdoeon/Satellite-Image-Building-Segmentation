import torch
import monai
import torch.nn as nn
import segmentation_models_pytorch as smp

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = monai.losses.DiceLoss()
        
    def forward(self, inputs, targets):
        return 0.8 * self.bce(inputs, targets) + 0.2 * self.dice(inputs, targets)
    
    
class LovaszBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszBCELoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.lovasz = smp.losses.LovaszLoss(mode='binary', per_image=True)
        
    def forward(self, inputs, targets):
        return 0.8 * self.bce(inputs, targets) + 0.2 * self.lovasz(inputs, targets)
    
class LovaszBCELoss2(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszBCELoss2, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.5, 1.0]))
        self.lovasz = smp.losses.LovaszLoss(mode='multiclass', per_image=True)
        
    def forward(self, inputs, targets):
        return 0.8 * self.ce(inputs, targets.long()) + 0.2 * self.lovasz(inputs, targets)    


def get_loss_function(name: str):
    if name == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss
    
    if name == 'DiceCELoss':
        return monai.losses.DiceCELoss
    
    if name == 'DiceBCELoss':
        return DiceBCELoss
    
    if name == 'LovaszBCELoss':
        return LovaszBCELoss

    if name == 'LovaszBCELoss2':
        return LovaszBCELoss2
