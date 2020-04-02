#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @Contact  : huangsy1314@163.com
# @Website  : https://huangshiyu13.github.io
# @File    : test_image.py

# 测试图片
import sys
from myutils import check_file
from PIL import Image
import numpy as np
import torch
from params import padding_image, img_mean, img_std, resize, DeepFakeModel
from timm.models import create_deepfake_model


def test_img(model_path, img_files):
    assert all(check_file(img_file) for img_file in img_files), 'file not exist!'

    use_cuda = True
    use_half = True
    print('To load model from {}'.format(model_path))
    model = create_deepfake_model(
        'efficientnet_deepfake',
        num_classes=2,
        in_chans=12,
        checkpoint_path=model_path,
        strict=False)
    print('Model loaded!')
    model = DeepFakeModel(model)
    if use_cuda:
        model.cuda()
        if use_half:
            model.half()
    model.eval()
    for img_file in img_files:
        img = np.transpose(padding_image(resize(np.array(Image.open(img_file).convert('RGB'), np.uint8))), (2, 0, 1))
        img = torch.from_numpy(img).float()
        img = img.sub_(img_mean).div_(img_std)
        if use_cuda:
            img = img.cuda()
            if use_half:
                img = img.half()
        img = [torch.cat([img, img.clone(), img.clone(), img.clone()], dim=0)]
        with torch.no_grad():
            scores = model(torch.stack(img, dim=0))
        scores = scores.cpu().numpy()[:, 0].tolist()
        print('{}\'s fake score:{}'.format(img_file, scores[0]))


def debug_test_img(model_path, img_files):
    print('To load model from {}'.format('model_path'))
    create_deepfake_model(
        'efficientnet_deepfake',
        num_classes=2,
        in_chans=12,
        strict=False)
    print('Model loaded!')


if __name__ == '__main__':
    model_path = './models/model_half.pth.tar'
    if len(sys.argv) <= 1:
        print('Please input your images. e.g. python test_images.py image_path1 image_path2')
        exit()
    img_files = sys.argv[1:]
    test_img(model_path, img_files)
