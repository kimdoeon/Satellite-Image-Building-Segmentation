import os
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

global num

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def divide_img(filename, dir, image, mask, stride=200, size=224, real_size=512):
    global num
    for a in range(0, 1024-size+1, stride):
        for b in range(0, 1024-size+1, stride):
            image_resized = image[a:a+size, b:b+size, :]
            mask_resized = mask[a:a+size, b:b+size]

            if size != real_size:
                image_resized = cv2.resize(image_resized, (real_size, real_size))
                mask_resized = cv2.resize(mask_resized, (real_size, real_size))

            cv2.imwrite(os.path.join(dir, 'images', f'{filename.replace(".png", "")}_{num}.png'), image_resized)
            cv2.imwrite(os.path.join(dir, 'masks', f'{filename.replace(".png", "")}_{num}.png'), mask_resized)
            num += 1

result = 'data_512'
os.makedirs(result, exist_ok=True)

img_dir = os.path.join(result, 'images')
os.makedirs(img_dir, exist_ok=True)

mask_dir = os.path.join(result, 'masks')
os.makedirs(mask_dir, exist_ok=True)

df = pd.read_csv('./train2.csv')

for img_path, mask_rle in zip(tqdm(df['img_path']), df['mask_rle']):
    filename = os.path.basename(img_path)
    image = cv2.imread(img_path)
    mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

    num = 0
    divide_img(filename=os.path.basename(img_path), 
               dir=result,
               image=image, 
               mask=mask, 
               stride=256, 
               size=512,
               real_size=512)