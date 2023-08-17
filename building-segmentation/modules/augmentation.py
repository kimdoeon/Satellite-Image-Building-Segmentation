import cv2
import torch
import random
import numpy as np
from albumentations.core.transforms_interface import DualTransform


def vmirrorup(img: np.ndarray) -> np.ndarray:
    axis = img.shape[1] // 2
    half = img[:axis, :, ...]
    half_flip = cv2.flip(half, 0)
    
    return np.concatenate((half, half_flip), axis=0)

def vmirrordown(img: np.ndarray) -> np.ndarray:
    axis = img.shape[1] // 2
    half = img[axis:, :, ...]
    half_flip = cv2.flip(half, 0)
    
    return np.concatenate((half_flip, half), axis=0)
    
def hmirrorup(img: np.ndarray) -> np.ndarray:
    axis = img.shape[0] // 2
    half = img[:, :axis, ...]
    half_flip = cv2.flip(half, 1)

    return np.concatenate((half, half_flip), axis=1)

def hmirrordown(img: np.ndarray) -> np.ndarray:
    axis = img.shape[0] // 2
    half = img[:, axis:, ...]
    half_flip = cv2.flip(half, 1)

    return np.concatenate((half_flip, half), axis=1)

class VerticalMirrorUp(DualTransform):
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return vmirrorup(img)
        
    def get_transform_init_args_names(self):
        return ()

class VerticalMirrorDown(DualTransform):
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return vmirrordown(img)
        
    def get_transform_init_args_names(self):
        return ()

class HorizontalMirrorUp(DualTransform):
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return hmirrorup(img)
    
    def get_transform_init_args_names(self):
        return ()
    
class HorizontalMirrorDown(DualTransform):
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return hmirrordown(img)
    
    def get_transform_init_args_names(self):
        return ()
    
def ricap(images, masks, output_size=2, beta=0.3):
    batch_size, channels, height, width = images.size()

    output_images = torch.zeros((output_size, channels, height, width), dtype=torch.uint8)
    output_masks = torch.zeros((output_size, height, width), dtype=torch.uint8)

    ratio = np.random.beta(beta, beta, size=output_size)
    ratio = np.clip(ratio, 0.3, 0.7)

    w = (width * ratio).astype(np.uint8)
    h = (height * ratio).astype(np.uint8)

    wx = np.random.randint(width - w + 1, size=output_size)
    hx = np.random.randint(height - h + 1, size=output_size)

    for i in range(output_size):
        indices = [j for j in range(batch_size) if j != i]
        crops_indices = random.sample(indices, 3)

        output_images[i, :, :hx[i], :wx[i]] = images[i, :, :hx[i], :wx[i]]
        output_images[i, :, :hx[i], wx[i]:] = images[crops_indices[0], :, :hx[i], wx[i]:]
        output_images[i, :, hx[i]:, :wx[i]] = images[crops_indices[1], :, hx[i]:, :wx[i]]
        output_images[i, :, hx[i]:, wx[i]:] = images[crops_indices[2], :, hx[i]:, wx[i]:]

        output_masks[i, :hx[i], :wx[i]] = masks[i, :hx[i], :wx[i]]
        output_masks[i, :hx[i], wx[i]:] = masks[crops_indices[0], :hx[i], wx[i]:]
        output_masks[i, hx[i]:, :wx[i]] = masks[crops_indices[1], hx[i]:, :wx[i]]
        output_masks[i, hx[i]:, wx[i]:] = masks[crops_indices[2], hx[i]:, wx[i]:]

    images = torch.cat((images, output_images))
    masks = torch.cat((masks, output_masks))

    return images, masks

def ricap2(images, masks, alpha, p):
    if np.random.rand() < p:
        I_x, I_y = images.size()[2:]

        w = int(np.round(I_x * np.random.beta(alpha, alpha)))
        h = int(np.round(I_y * np.random.beta(alpha, alpha)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]

        cropped_images = {}
        cropped_masks = {}
        for k in range(4):
            idx = torch.randperm(images.size(0))
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            cropped_images[k] = images[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
            cropped_masks[k] = masks[idx][:, x_k:x_k + w_[k], y_k:y_k + h_[k]]

        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
            torch.cat((cropped_images[2], cropped_images[3]), 2)),
        3)
        patched_masks = torch.cat(
            (torch.cat((cropped_masks[0], cropped_masks[1]), 1),
            torch.cat((cropped_masks[2], cropped_masks[3]), 1)),
        2)

        return patched_images, patched_masks
    
    return images, masks