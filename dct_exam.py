import matplotlib
#matplotlib.use('TkAgg')  # Must be before pyplot
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from scipy.fftpack import dct, idct

# DCT and IDCT functions
def dct2(x, axis1=-2, axis2=-1):
    x_np = x.detach().cpu().numpy()
    x_dct = dct(dct(x_np, axis=axis2, norm='ortho'), axis=axis1, norm='ortho')
    return torch.tensor(x_dct, dtype=x.dtype).to(x.device)

def idct2(x, axis1=-2, axis2=-1):
    x_np = x.detach().cpu().numpy()
    x_idct = idct(idct(x_np, axis=axis2, norm='ortho'), axis=axis1, norm='ortho')
    return torch.tensor(x_idct, dtype=x.dtype).to(x.device)

# Load image
img = cv2.imread("example_img.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("⚠️ 'example_img.jpg' not found. Please check the path.")
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32) / 255.0

# Convert to tensor
img_tensor = torch.tensor(img)

# DCT and IDCT
dct_img = dct2(img_tensor)
idct_img = idct2(dct_img).clamp(0, 1)

dct_vis = torch.log(torch.abs(dct_img) + 1e-5)
dct_vis = dct_vis - dct_vis.min()
dct_vis = dct_vis / dct_vis.max()
dct_vis = dct_vis ** 0.5  # gamma correction to brighten mid-range

# Plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_tensor.numpy(), cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Focus only on top-left 64x64 block
plt.figure()
plt.imshow(dct_vis[:64, :64].numpy(), cmap='gray')
plt.title("Top-left corner of DCT")
plt.axis('off')
plt.show()

plt.subplot(1, 3, 3)
plt.imshow(idct_img.numpy(), cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')

plt.tight_layout()
plt.show()
