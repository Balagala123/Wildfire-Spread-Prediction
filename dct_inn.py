#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 13:03:15 2025

@author: sowmyabalagala
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:41:28 2025

@author: Dell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import hadamard
from scipy.fftpack import dct, idct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hadamard transforms
def dct2(x, axis1=-2, axis2=-1):
    x_np = x.detach().cpu().numpy()
    x_dct = dct(dct(x_np, axis=axis2, norm='ortho'), axis=axis1, norm='ortho')
    return torch.tensor(x_dct, dtype=x.dtype).to(x.device)

def idct2(x, axis1=-2, axis2=-1):
    x_np = x.detach().cpu().numpy()
    x_idct = idct(idct(x_np, axis=axis2, norm='ortho'), axis=axis1, norm='ortho')
    return torch.tensor(x_idct, dtype=x.dtype).to(x.device)

# Thresholding module
class Thresholding(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.T = nn.Parameter(torch.rand(shape) / 10)

    def forward(self, x):
        return torch.copysign(F.relu(torch.abs(x) - torch.abs(self.T)), x)

class HadamardUnet(nn.Module):
    def __init__(self, input_channels=12, input_size=64, output_channels=1, dropout_rate=0.3):
        super(HadamardUnet, self).__init__()
        self.height = input_size // 2
        self.width = input_size // 2

        self.conv1 = nn.Conv2d(input_channels, 4, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(4, 4, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(4)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.deconv1 = nn.ConvTranspose2d(4, 4, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm2d(4)
        self.dropout3 = nn.Dropout2d(dropout_rate)

        self.deconv2 = nn.ConvTranspose2d(4, output_channels, kernel_size=4, stride=2, padding=1)

        self.v1 = nn.Parameter(torch.rand((self.height, self.width)))
        self.v2 = nn.Parameter(torch.rand((self.height, self.width)))
        self.v3 = nn.Parameter(torch.rand((self.height, self.width)))

        self.ST1 = Thresholding((self.height, self.width))
        self.ST2 = Thresholding((self.height, self.width))
        self.ST3 = Thresholding((self.height, self.width))



    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.dropout1(x1)
    
        x2 = dct(dct(x1, axis=-1), axis=-2)
        x3 = self.v1 * x2
        x4 = self.ST1(x3)
        x5 = idct(idct(x4, axis=-1), axis=-2)
    
        x6 = F.relu(self.bn2(self.conv2(x5)))
        x6 = self.dropout2(x6)
    
        x7 = dct(dct(x6, axis=-1), axis=-2)
        x8 = self.v2 * x7
        x9 = self.ST2(x8)
        x10 = idct(idct(x9, axis=-1), axis=-2)
    
        x11 = F.relu(self.bn3(self.deconv1(x10)))
        x11 = self.dropout3(x11)
        x11 = x11 + x6  # Skip connection
    
        x12 = dct(dct(x11, axis=-1), axis=-2)
        x13 = self.v3 * x12
        x14 = self.ST3(x13) + x4  # Residual
        x15 = idct(idct(x14, axis=-1), axis=-2)
    
        x_out = self.deconv2(x15)
        return x_out
