from modules.utils import load_yaml
from modules.optimizer import get_optimizer
from modules.scheduler import get_scheduler
from modules.loss import get_loss_function
from modules.model import get_smp_model
from modules.dataset import smpDataset
from modules.augmentation import *

import os
import random
import shutil
import argparse
from glob import glob
from tqdm import tqdm
from datetime import datetime

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='train.yaml')
args = parser.parse_args()

prj_dir = './'
config_path = os.path.join(prj_dir, 'config', args.config)
config = load_yaml(config_path)

train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")

train_result_dir = os.path.join(prj_dir, 'results', 'train', train_serial)
os.makedirs(train_result_dir, exist_ok=True)
os.makedirs(f'{train_result_dir}/images/train', exist_ok=True)
os.makedirs(f'{train_result_dir}/images/val', exist_ok=True)
os.makedirs(f'{train_result_dir}/weights', exist_ok=True)

shutil.copy(config_path, os.path.join(train_result_dir, 'train.yaml'))

torch.cuda.manual_seed(config['seed'])
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = A.Compose([
    A.Rotate(limit=90, p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),

    A.OneOf([
        VerticalMirrorUp(1),
        VerticalMirrorDown(1),
    ], p=0.5),

    A.OneOf([
        HorizontalMirrorUp(1),
        HorizontalMirrorDown(1),
    ], p=0.5),

    A.OneOf([
        A.Blur(p=1),
        A.AdvancedBlur(p=1),
        A.MotionBlur(p=1),
    ], p=0.6),

    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        A.RandomShadow(p=1),
        A.CLAHE(p=1),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, always_apply=False, p=1),
        A.Equalize(p=1),
    ], p=0.5),
    
    A.RandomCrop(config['input_size'], config['input_size']),
    A.Normalize(),
    ToTensorV2(),
])

train_images = sorted(glob('./data/data_512/images/*.png'))
train_masks = [img.replace('images', 'masks') for img in train_images]

print('train len:', len(train_images))

oba_images = glob('./data/oba/result2/images/*.png')
oba_masks = [img.replace('images', 'masks') for img in oba_images]

print(f'oba len: {len(oba_images)}')

train_dataset = smpDataset(train_images + oba_images, 
                           train_masks + oba_masks, 
                           train_transform, 
                           infer=False)

train_dataloader = DataLoader(train_dataset, 
                              batch_size=config['batch_size'], 
                              num_workers=config['num_workers'],
                              shuffle=True)

model = get_smp_model(name=config['model']['architecture'])

model = model(encoder_name=config['model']['encoder'],
              encoder_weights=config['model']['encoder_weight'],
              in_channels=config['model']['in_channel'],
              classes=config['model']['n_classes'],
).to(device)

if config['model']['pretrained'] != False:
    weights = torch.load(config['model']['pretrained'])
    model.load_state_dict(weights['model'])

optimizer = get_optimizer(name=config['optimizer']['name'])
optimizer = optimizer(model.parameters(), **config['optimizer']['args'])

start_epoch = 0
if config['model']['pretrained'] != False:
    optimizer.load_state_dict(weights['optimizer'])
    start_epoch = weights['epochs']

scheduler = get_scheduler(name=config['scheduler']['name'])
scheduler = scheduler(optimizer=optimizer, **config['scheduler']['args'])

loss_func = get_loss_function(name=config['loss']['name'])
loss_func = loss_func()

scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter(f'{train_result_dir}/tensorboard')

def run_batch_train(images, labels):
    images = images.to(device)
    labels = labels.float().to(device)
    
    optimizer.zero_grad()
    
    if config['amp']:
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_func(outputs, labels.unsqueeze(1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    else:
        outputs = model(images)
        
        loss = loss_func(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # scheduler.step()

    return loss, outputs
    
print('training start')

for epoch in range(start_epoch, config['n_epochs']):
    train_losses = []
    model.train()
    for idx, (images, labels) in enumerate(tqdm(train_dataloader)):
        images_, labels_ = ricap2(images, labels, 0.3, 0.5)

        loss, predicts = run_batch_train(images_, labels_)
        train_losses.append(loss.item())

    writer.add_scalar('Loss/train', np.mean(train_losses), epoch)

    log = f'[{epoch}/{config["n_epochs"]}] train loss: {np.mean(train_losses):.4f}'
    print(log)

    state = {
        'epochs': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, f'{train_result_dir}/weights/last.pth')
    
writer.close()