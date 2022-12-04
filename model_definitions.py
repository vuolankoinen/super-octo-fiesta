import torch
import torchvision


def ResNet18(
    cats=3,
):
    model = torchvision.models.resnet18(pretrained=False)

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=512, out_features=256),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features=256, out_features=128),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=cats),
    )

    return model


def EfficientNetb4(
    cats=3,
):
    model = torchvision.models.efficientnet.efficientnet_b4(pretrained=False)

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=1792, out_features=625),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features=625, out_features=256),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=256, out_features=cats),
    )

    return model
