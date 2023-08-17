from modules.utils import load_yaml, rle_decode, rle_encode, draw_fig
from modules.model import get_transformers_model
from modules.optimizer import get_optimizer
from modules.scheduler import get_scheduler
from modules.loss import get_loss_function

from modules.dataset import transformsCustomDataset
from modules.augmentation import *

import os
import random
import shutil
from glob import glob
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from transformers import (
    AutoImageProcessor,
    TrainingArguments, 
    Trainer,
    SegformerConfig
)

prj_dir = './'
config_path = os.path.join(prj_dir, 'config', 'train_seg_512.yaml')
config = load_yaml(config_path)

# Set train serial: ex) 20211004
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
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
        A.RandomBrightnessContrast(p=1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        A.RandomShadow(p=1),
        A.CLAHE(p=1),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, always_apply=False, p=1),
        A.Equalize(p=1),
    ], p=0.5),
    
    A.RandomCrop(config['input_size'], config['input_size']),
])
val_transfrom = A.Compose([
])

trian_images = sorted(glob(f"{config['data_dir']}/train/images/*.png"))
trian_masks = [image.replace("images", "masks") for image in trian_images]

val_images = sorted(glob(f"{config['data_dir']}/val/images/*.png"))
val_masks = [image.replace("images", "masks") for image in val_images]

print('train len:', len(trian_images), 'test len:', len(val_images))

processor = AutoImageProcessor.from_pretrained(config['processor'])

train_dataset = transformsCustomDataset(processor, 
                                        trian_images, 
                                        trian_masks, 
                                        train_transform, 
                                        infer=False)

val_dataset = transformsCustomDataset(processor, 
                                      val_images, 
                                      val_masks, 
                                      val_transfrom, 
                                      infer=False)

train_dataloader = DataLoader(train_dataset, 
                              batch_size=config['batch_size'], 
                              num_workers=config['num_workers'],
                              shuffle=True)

val_dataloader = DataLoader(val_dataset, 
                            batch_size=config['batch_size'], 
                            num_workers=config['num_workers'],
                            shuffle=False)

model = get_transformers_model(name=config['model']['name'])
model = model.from_pretrained(
    config['model']['weight'],
    num_labels=1,
    ignore_mismatched_sizes=True
).to(device)
print('model')

optimizer = get_optimizer(name=config['optimizer']['name'])
optimizer = optimizer(model.parameters(), **config['optimizer']['args'])

scheduler = get_scheduler(name=config['scheduler']['name'])
scheduler = scheduler(optimizer=optimizer, **config['scheduler']['args'])

loss_func = get_loss_function(name=config['loss']['name'])
loss_func = loss_func()

scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter(f'{train_result_dir}/tensorboard')

def upscale_logits(logit_outputs, res=224):
    return nn.functional.interpolate(
        logit_outputs,
        size=(res, res),
        mode='bilinear',
        align_corners=False
    )

def run_batch_train(batch):
    images = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()
    
    if config['amp']:
        with torch.cuda.amp.autocast():
            outputs = model(pixel_values=images, labels=labels)
            upsampled_logits = upscale_logits(outputs.logits)

            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    else:
        outputs = model(pixel_values=images, labels=labels)
        upsampled_logits = upscale_logits(outputs.logits)
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # scheduler.step()

    return loss, upsampled_logits

def run_batch_val(batch):
    with torch.no_grad():
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=images, labels=labels)
        upsampled_logits = upscale_logits(outputs.logits)
        
        loss = outputs.loss

        return loss, upsampled_logits
    

print('training start')
best_loss = 10

for epoch in range(config['n_epochs']):
    train_losses = []
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        loss, predicts = run_batch_train(batch)
        train_losses.append(loss.item())
        
        if config['step'] != 0 and idx and idx % config['step'] == 0:
            writer.add_scalar('Loss/train', np.mean(train_losses), epoch*len(train_dataloader)+idx)
            print(f'train [{idx}] {config["loss"]["name"]}: {np.mean(train_losses):.4f}')

            seg_prob = torch.sigmoid(predicts).detach().cpu().numpy().squeeze()
            seg = (seg_prob > 0.5).astype(np.uint8)
            
            draw_fig(seg, batch['labels'], f'{train_result_dir}/images/train/{str(epoch)}_{str(idx)}.png', 4)
            torch.save(model.state_dict(), f'{train_result_dir}/weights/last.pth')
            train_losses = []
    
    print(f'train [{epoch}/{config["n_epochs"]}] {config["loss"]["name"]}: {np.mean(train_losses):.4f}')
    writer.add_scalar('Loss/train', np.mean(train_losses), epoch)

    val_losses = []
    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(tqdm(val_dataloader)):
            loss, predicts = run_batch_val(batch)
            val_losses.append(loss.item())
    
    print(f'[{epoch}/{config["n_epochs"]}] train loss: {np.mean(train_losses):.4f} val loss: : {np.mean(val_losses):.4f}')
    writer.add_scalar('Loss/val', np.mean(val_losses), epoch)
    
    if best_loss > np.mean(val_losses):
        best_loss = np.mean(val_losses)
        torch.save(model.state_dict(), f'{train_result_dir}/weights/best.pth')

    seg_prob = torch.sigmoid(predicts).detach().cpu().numpy().squeeze()
    seg = (seg_prob > 0.5).astype(np.uint8)

    draw_fig(seg, batch["labels"], f'{train_result_dir}/images/val/{str(epoch)}.png', 2)
    torch.save(model.state_dict(), f'{train_result_dir}/weights/last.pth')
    
writer.close()