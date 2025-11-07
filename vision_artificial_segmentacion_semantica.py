import torch
import wget
import zipfile
import os 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

input = torch.randn(64, 10, 20, 20)
# aumentamos la dimensión x2
conv_trans = torch.nn.ConvTranspose2d(
    in_channels=10, 
    out_channels=20, 
    kernel_size=2, 
    stride=2)
output = conv_trans(input)
print(output.shape)


url = 'https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/MRIs.zip'
wget.download(url)


with zipfile.ZipFile('MRIs.zip', 'r') as zip_ref:
    zip_ref.extractall('.')



path = Path('./MRIs')
imgs = [path/'MRIs'/i for i in os.listdir(path/'MRIs')]
ixs = [i.split('_')[-1] for i in os.listdir(path/'MRIs')]
masks = [path/'Segmentations'/f'segm_{ix}' for ix in ixs]

print(len(imgs), len(masks))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
img = np.load(imgs[0])
mask = np.load(masks[0])
ax1.imshow(img)
ax2.imshow(mask)
ax3.imshow(img)
ax3.imshow(mask, alpha=0.4)
plt.show()


print(img.shape, img.dtype, img.max(), img.min())

print(mask.shape, mask.dtype, mask.max(), mask.min())

# one-hot encoding
mask_oh = (np.arange(3) == mask[...,None]).astype(np.float32) 

print(mask_oh.shape, mask_oh.dtype, mask_oh.max(), mask_oh.min())