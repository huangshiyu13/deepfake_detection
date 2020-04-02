#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @Contact  : huangsy1314@163.com
# @Website  : https://huangshiyu13.github.io
# @File    : params.py
import os
import numpy as np
import torch
import torch.nn as nn
import cv2

img_mean = (0.485, 0.456, 0.406)
img_std = (0.229, 0.224, 0.225)
img_mean = torch.tensor([x * 255 for x in img_mean]).view(3, 1, 1)
img_std = torch.tensor([x * 255 for x in img_std]).view(3, 1, 1)
image_max_height = 600
image_max_width = 600
image_max_w_h = (image_max_width, image_max_height)
img_num = 4


class DeepFakeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.basemodel = model
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        logistic = self.basemodel(x)
        return self.softmax(logistic)


def resize(image):
    height_o, width_o = image.shape[0:2]

    if float(height_o) / width_o > float(image_max_w_h[1]) / image_max_w_h[0]:
        height_target = image_max_w_h[1]
        width_target = int(width_o * float(height_target) / height_o)
    else:
        width_target = image_max_w_h[0]
        height_target = int(height_o * float(width_target) / width_o)
    image = cv2.resize(image, (width_target, height_target))
    return image


def padding_image(image):
    height_o, width_o = image.shape[0:2]
    if height_o == image_max_height and width_o == image_max_width:
        return image
    h_pad_len_top = int((image_max_height - height_o) / 2)
    h_pad_len_bottom = image_max_height - height_o - h_pad_len_top
    w_pad_len_left = int((image_max_width - width_o) / 2)
    h_pad_len_right = image_max_width - width_o - w_pad_len_left

    return np.pad(image, ((h_pad_len_top, h_pad_len_bottom), (w_pad_len_left, h_pad_len_right), (0, 0)), 'constant',
                  constant_values=0)
