# This file sets up and performs a single experiment. Modify and save to 2.py, 3.py... to perform other experiments.

import sys
import torch
from torchsampler import ImbalancedDatasetSampler
import datetime
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(1, "./")
import config
import augmentations
import models
import lookahead
import trainer

torch.manual_seed(134)

model_name = f"koe{__name__}"

img_size = config.IMG_SIZE
batch_size = 4
learning_rate = 1e-3
weight_decay = 1e-2
epochs = 100

device = torch.device("cuda")

# Set up augmentations
train_transform = torchvision.transforms.Compose(
    [
        augmentations.WeatherAugmentations(p=0.3, output="pil"),
        augmentations.GeometricAugs(img_size=config.IMG_SIZE),
        augmentations.visual_transforms,
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

val_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        augmentations.AdaptivePad(size=config.IMG_SIZE),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Set up data
train_set = torchvision.datasets.ImageFolder(
    config.TRAIN_DIR,
    transform=train_transform,
)
print(f"Training images: {len(train_set)}")

val_set = torchvision.datasets.ImageFolder(config.VAL_DIR, transform=val_transform)
print(f"Validation images: {len(val_set)}")

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    # shuffle=True,
    sampler=ImbalancedDatasetSampler(train_set),
)
print(f"No. of batches in training loader: {len(train_loader)}")
print(f"No. of Total examples: {len(train_loader.dataset)}")
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
print(f"No. of batches in validation loader: {len(val_loader)}")
print(f"No. of Total examples: {len(val_loader.dataset)}")

# Set up model
model = models.ResNet18(cats=len(config.CATEGORIES))
model.to(device)

# Set up training

base_opt = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
    amsgrad=False,
)
optimizer = lookahead.Lookahead(base_opt, k=5, alpha=0.3)

criterion = nn.CrossEntropyLoss()

trainer_instance = trainer.Trainer(criterion, optimizer)
trainer_instance.fit(model, train_loader, val_loader, epochs=epochs, name=model_name)
