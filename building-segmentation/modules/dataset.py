import os
import cv2
from torch.utils.data import Dataset


class transformsCustomDataset(Dataset):
    def __init__(self, processor, images, masks, transform, infer=False):
        self.images = images
        self.masks = masks
        self.processor = processor
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']

            inputs = self.processor(image, return_tensors='pt')
            inputs = {k:v.squeeze(0) for k, v in inputs.items()}
            return inputs, os.path.basename(self.images[idx])
        
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        inputs = self.processor(image, return_tensors='pt')
        inputs = {k:v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = mask

        return inputs
    
    
class smpDataset(Dataset):
    def __init__(self, images, masks, transform, infer=False):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented['image']

            return image, os.path.basename(self.images[idx])
        
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask