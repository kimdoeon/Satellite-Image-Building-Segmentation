from transformers import (
    BeitForSemanticSegmentation,
    SegformerForSemanticSegmentation, 
)

import segmentation_models_pytorch as smp

def get_transformers_model(name: str):
    if name == 'beit':
        return BeitForSemanticSegmentation
    
    if name == 'segformer':
        return SegformerForSemanticSegmentation

def get_smp_model(name: str):
    if name == 'DeepLabV3Plus':
        return smp.DeepLabV3Plus
    
    if name == 'UnetPlusPlus':
        return smp.UnetPlusPlus