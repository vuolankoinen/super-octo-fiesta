import math
import cv2
import imgaug
import torchvision
import PIL.Image
import numpy as np
import torch

class AdaptiveCrop(object):
    def __init__(self, frac=0.8):
        self.fraction = frac

    def __call__(self, sample):
        return torchvision.transforms.RandomCrop(
            size=(
                int(self.fraction * sample.size[1]),
                int(self.fraction * sample.size[0]),
            )
        )(sample)


class AdaptivePad(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if isinstance(sample, PIL.Image.Image):
            return torchvision.transforms.Pad(
                padding=(
                    0,
                    0,
                    int(self.size[1] - sample.size[0]),
                    int(self.size[0] - sample.size[1]),
                )
            )(sample)
        elif isinstance(sample, torch.Tensor):
            return torchvision.transforms.Pad(
                padding=(
                    0,
                    0,
                    int(self.size[1] - sample.shape[2]),
                    int(self.size[0] - sample.shape[1]),
                )
            )(sample)


class GeometricAugs(torchvision.transforms.transforms.Compose):
    def __init__(self, img_size=None):
        super().__init__(
            [
                AdaptiveCrop(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomErasing(
                    p=0.1, value="random", scale=(0.0, 0.20)
                ),
                torchvision.transforms.RandomErasing(  # "street light pole"
                    p=0.1, value=(0.6, 0.3, 0.02), scale=(0.1, 0.25), ratio=(5, 9)
                ),
                torchvision.transforms.RandomRotation(degrees=(0, 10)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
            ]
            + (  # Resize (by padding) if img_size is required
                [AdaptivePad(size=(img_size[0], img_size[1]))]
                if img_size is not None
                else []
            )
        )


visual_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomApply(
            transforms=[
                torchvision.transforms.ColorJitter(
                    brightness=0.6
                ),  # Värit olennaisia, joten hue vain hämäisi
            ],
            p=0.3,
        ),
        torchvision.transforms.RandomApply(
            transforms=[
                torchvision.transforms.ColorJitter(contrast=0.4),
            ],
            p=0.3,
        ),
        torchvision.transforms.RandomApply(
            transforms=[
                torchvision.transforms.ColorJitter(saturation=0.5),
            ],
            p=0.3,
        ),
        torchvision.transforms.RandomApply(
            transforms=[
                torchvision.transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5)),
            ],
            p=0.2,
        ),
        torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
        torchvision.transforms.RandomApply(
          transforms=[
              AddGaussianNoise(0.1, 0.05),
          ],
          p=0.5,
        )
    ]
)
