import os
import cv2
from tqdm.auto import tqdm

kfold = 0

with open(os.path.join('./data/kfold', f'kfold_{kfold}/val.txt'), 'r') as f:
    val_list = [line.strip() for line in f]

val_images, val_masks = [], []
for filename in tqdm(val_list):
    val_images.append(f'./data/train_img/{filename}.png')
    val_masks.append(f'./data/train_mask/{filename}.png')

val_img_dir = './data/val_img'
val_mask_dir = './data/val_mask'

os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

global num

def divide_img(filename, image, mask, stride=200, size=224, real_size=512):
    global num
    for a in range(0, 1024-size+1, stride):
        for b in range(0, 1024-size+1, stride):
            image_resized = image[a:a+size, b:b+size, :]
            mask_resized = mask[a:a+size, b:b+size]

            if size != real_size:
                image_resized = cv2.resize(image_resized, (real_size, real_size))
                mask_resized = cv2.resize(mask_resized, (real_size, real_size))

            cv2.imwrite(os.path.join(val_img_dir, f'{filename.replace(".png", "")}_{num}.png'), image_resized)
            cv2.imwrite(os.path.join(val_mask_dir, f'{filename.replace(".png", "")}_{num}.png'), mask_resized)
            num += 1

for img_path, mask_path in zip(tqdm(val_images), val_masks):
    filename = os.path.basename(img_path)
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    num = 0
    divide_img(filename=os.path.basename(img_path), 
               image=image, 
               mask=mask, 
               stride=200, 
               size=224,
               real_size=224)