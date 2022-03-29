import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


'''
# test
train_mask = pd.read_csv('../tcdata/train_mask.csv', sep='\t', names=['name', 'mask'])

img = cv2.imread('../tcdata/train/' + train_mask['name'].iloc[0])

mask = rle_decode(train_mask['mask'].iloc[0])  # 把标签转化为图像


# 展示一下图片
plt.figure('image')
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.savefig('test.jpg')

print(rle_encode(mask) == train_mask['mask'].iloc[0])  # 验证转化前后是否相同
'''


