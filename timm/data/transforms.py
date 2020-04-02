import torch
import torchvision.transforms.functional as F
from PIL import Image
import warnings
import math
import random
import numpy as np
import numbers
from PIL import ImageFilter


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class MultiToNumpy:
    def __call__(self, pil_imgs):
        np_imgs = [np.array(pil_img, dtype=np.uint8) for pil_img in pil_imgs]
        if np_imgs[0].ndim < 3:
            np_imgs = [np.expand_dims(np_img, axis=-1) for np_img in np_imgs]
        np_imgs = [np.rollaxis(np_img, 2) for np_img in np_imgs]  # HWC to CHW
        return np_imgs


class MultiConcate:
    def __call__(self, np_imgs):
        np_imgs_concat = np.concatenate(np_imgs, axis=0)
        return np_imgs_concat


class ToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype)


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomResize():
    """Resize the given PIL Image randomly.
        Args:
            scale: range of size of the origin size cropped
            interpolation: Default: PIL.Image.BILINEAR
        """

    def __init__(self, scale=(0.9, 1.1),
                 interpolation='bilinear'):
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        random_scale = random.uniform(self.scale[0], self.scale[1])
        w = img.size[0]
        h = img.size[1]
        target_size = [int(h * random_scale), int(w * random_scale)]
        return F.resize(img, target_size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(scale={0}'.format(self.scale)
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class MultiRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return [F.hflip(img) for img in imgs]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class MultiBlur(object):
    def __init__(self, p, blur_radiu):
        self.p = p
        self.blur_radiu = blur_radiu

    def __call__(self, imgs):
        re = []
        for img in imgs:
            if random.random() < self.p:
                re.append(img.filter(ImageFilter.GaussianBlur(radius=self.blur_radiu)))
            else:
                re.append(img)
        return re

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, radiu={})'.format(self.rotate_range, self.blur_radiu)


class MultiRotate(object):
    def __init__(self, rotate_range):
        self.rotate_range = rotate_range

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        rotate_degree = random.randint(-self.rotate_range, self.rotate_range)

        return [img.rotate(rotate_degree, expand=True) for img in imgs]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.rotate_range)


class MultiRandomResize(RandomResize):
    """Resize the given PIL Image randomly.
        Args:
            scale: range of size of the origin size cropped
            interpolation: Default: PIL.Image.BILINEAR
        """

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        random_scale = random.uniform(self.scale[0], self.scale[1])
        w = imgs[0].size[0]
        h = imgs[0].size[1]
        target_size = [int(h * random_scale), int(w * random_scale)]
        return [F.resize(img, target_size, interpolation) for img in imgs]


from torchvision.transforms import ColorJitter, RandomCrop


class MultiRandomCrop(RandomCrop):
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            imgs = [F.pad(img, self.padding, self.fill, self.padding_mode) for img in imgs]

        # pad the width if needed
        if self.pad_if_needed and imgs[0].size[0] < self.size[1]:
            imgs = [F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode) for img in imgs]
        # pad the height if needed
        if self.pad_if_needed and imgs[0].size[1] < self.size[0]:
            imgs = [F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode) for img in imgs]
        i, j, h, w = self.get_params(imgs[0], self.size)
        return [F.crop(img, i, j, h, w) for img in imgs]


class MultiColorJitter(ColorJitter):
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return [transform(img) for img in imgs]


class MultiFlicker:
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, imgs):
        img_size = imgs[0].size
        return [Image.new('RGB', img_size[:2]) if random.random() < self.probability else img for img in imgs]
