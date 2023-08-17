import numpy as np
import yaml
import matplotlib.pyplot as plt

def save_yaml(path, obj):
	with open(path, 'w') as f:
		yaml.dump(obj, f, sort_keys=False)
		

def load_yaml(path):
	with open(path, 'r') as f:
		return yaml.load(f, Loader=yaml.FullLoader)


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


def draw_fig(images, labels, filename, count=4):
    fig, axes = plt.subplots(count, 2, figsize=(5, 5))
    for i in range(count):
        axes[i][0].imshow(images[i])
        axes[i][1].imshow(labels[i])

    fig.savefig(filename)
    plt.close(fig)