# Deepfake Detection

## Description

Use deep models to detect fake images. 

We trained an EfficientNet model to detect deepfake images. 
Check out this [repo](https://github.com/EndlessSora/DeeperForensics-1.0) 
to know more about deepfake images.

- Real image: 

![](./test_images/0.png)

- Fake image:

![](./test_images/1.jpeg)

## Installation

`pip install -r requirements.txt`

## Usage

- ### test images using trained model

You need download the pretrained model from BaiduYun:

download link: `https://pan.baidu.com/s/16NIV5BVUITKwolQzPbj9Zw`  password:`r3i8`

And put the model(`model_half.pth.tar`) 
under `./models`. Then you can run python script as below:

`python test_images.py ./test_images/0.png ./test_images/1.jpeg`

- ### train on your own dataset

The training code is to be released.

## Cite
Please cite our code if you use this code or our models in your own work:
```
@misc{deepfake_detection,
  title={Deepfake Detection},
  author={Huang, Shiyu and others},
  howpublished={\url{https://github.com/huangshiyu13/deepfake_detection}},
  year={2020}
}
```
## Author

[Shiyu Huang](https://huangshiyu13.github.io/) (huangsy1314@163.com)

If you want to train or test your dataset using this code, 
please feel free to contact me.

## Acknowledgement

This code is based on https://github.com/rwightman/pytorch-image-models


