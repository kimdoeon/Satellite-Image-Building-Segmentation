from modules.utils import load_yaml, rle_decode, rle_encode, draw_fig
from modules.optimizer import get_optimizer
from modules.scheduler import get_scheduler
from modules.loss import get_loss_function
from modules.model import get_transformers_model, get_smp_model
from modules.dataset import transformsCustomDataset, smpDataset
from modules.augmentation import *

import os
import random
import shutil
import argparse
from glob import glob
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import slack_sdk

def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='train.yaml')
args = parser.parse_args()

prj_dir = './'
config_path = os.path.join(prj_dir, 'config', args.config)
config = load_yaml(config_path)

client = slack_sdk.WebClient(token=config["slack"]["token"])

client.chat_postMessage(
    channel=config["slack"]["channel"],
    text=f'{config["name"]} start'
)

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
val_transfrom = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

with open(os.path.join('./data/kfold', f'kfold_{config["kfold"]}/train.txt'), 'r') as f:
    train_list = [line.strip() for line in f]

train_images, train_masks = [], []
for filename in tqdm(train_list):
    train_images.append(os.path.join(config['data_dir'], f'images/{filename}.png'))
    train_masks.append(os.path.join(config['data_dir'], f'masks/{filename}.png'))

val_images = sorted(glob('./data/val_img/*.png'))
val_masks = [img.replace('val_img', 'val_mask') for img in val_images]

print('train len:', len(train_images), 'test len:', len(val_images))

oba_images = glob('./data/oba/result/images/*.png')
oba_masks = [img.replace('images', 'masks') for img in oba_images]

print(f'oba train len: {len(oba_images)}, val len: {len(oba_masks)}')

train_dataset = smpDataset(train_images + oba_images, 
                           train_masks + oba_masks, 
                           train_transform, 
                           infer=False)

val_dataset = smpDataset(val_images, 
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

model = get_smp_model(name=config['model']['architecture'])

model = model(encoder_name=config['model']['encoder'],
              encoder_weights=config['model']['encoder_weight'],
              in_channels=config['model']['in_channel'],
              classes=config['model']['n_classes'],
).to(device)

optimizer = get_optimizer(name=config['optimizer']['name'])
optimizer = optimizer(model.parameters(), **config['optimizer']['args'])

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
            if np.isinf(loss.item()) or np.isnan(loss.item()):
                client.chat_postMessage(
                    config["slack"]["channel"],
                    text=f'{config["name"]} - NaN or Inf'
                )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    else:
        outputs = model(images)
        
        loss = loss_func(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

    scheduler.step()

    return loss, outputs

def run_batch_val(images, labels):
    with torch.no_grad():
        images = images.to(device)
        labels = labels.float().to(device)
        
        if config['amp']:
            with torch.cuda.amp.autocast():
                outputs = model(images)

                loss = loss_func(outputs, labels.unsqueeze(1))
                if np.isinf(loss.item()) or np.isnan(loss.item()):
                    client.chat_postMessage(
                        config["slack"]["channel"],
                        text=f'{config["name"]} - NaN or Inf'
                    )
        else:
            outputs = model(images)
            
            loss = loss_func(outputs, labels.unsqueeze(1))

        return loss, outputs
    
print('training start')
best_score = 0

for epoch in range(config['n_epochs']):
    train_losses = []
    model.train()
    for idx, (images, labels) in enumerate(tqdm(train_dataloader)):
        images_, labels_ = ricap2(images, labels, 0.3, 0.5)

        loss, predicts = run_batch_train(images_, labels_)
        train_losses.append(loss.item())
    
        if config['step']['isthatrue'] and idx % config['step']['idx'] == 0:
            writer.add_scalar('Loss/train', np.mean(train_losses), epoch*len(train_dataloader)+idx)
            print(f'train [{idx}] {config["loss"]["name"]}: {np.mean(train_losses)}')

            seg_prob = torch.sigmoid(predicts).detach().cpu().numpy().squeeze()
            seg = (seg_prob > 0.5).astype(np.uint8)

            draw_fig(seg, labels, f'{train_result_dir}/images/train/{str(epoch)}_{str(idx)}.png', 4)
            torch.save(model.state_dict(), f'{train_result_dir}/weights/last.pth')
            train_losses = []

    val_losses, score1_list, score2_list = [], [], []
    model.eval()
    for idx, (images, labels) in enumerate(tqdm(val_dataloader)):
        loss, predicts = run_batch_val(images, labels)
        seg_prob = torch.sigmoid(predicts).detach().cpu().numpy().squeeze()
        seg1 = (seg_prob > 0.35).astype(np.uint8)
        seg2 = (seg_prob > 0.5).astype(np.uint8)

        val_losses.append(loss.item())
        score1_list.append(dice_score(seg1, labels.detach().cpu().numpy()))
        score2_list.append(dice_score(seg2, labels.detach().cpu().numpy()))

    score1 = np.mean(score1_list)
    score2 = np.mean(score2_list)
    
    writer.add_scalar('Loss/train', np.mean(train_losses), epoch)
    writer.add_scalar('Loss/val', np.mean(val_losses), epoch)
    writer.add_scalar('score/0.35', score1, epoch)
    writer.add_scalar('score/0.5', score2, epoch)

    log = f'[{epoch}/{config["n_epochs"]}] train loss: {np.mean(train_losses):.4f}, val loss: : {np.mean(val_losses):.4f}, dice035: {score1:.4f}, dice05: {score2:.4f}'
    print(log)
    
    client.chat_postMessage(
        config["slack"]["channel"],
        text=f'{config["name"]} {log}'
    )

    state = {
        'epochs': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, f'{train_result_dir}/weights/last.pth')

    if best_score < score1 or best_score < score2:
        if score1 < score2:
            best_score = score2
        else:
            best_score = score1

        torch.save(state, f'{train_result_dir}/weights/best.pth')

    draw_fig(seg2, labels, f'{train_result_dir}/images/val/{str(epoch)}.png', 2)
    
writer.close()