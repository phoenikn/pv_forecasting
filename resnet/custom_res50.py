import torch.nn as nn
import torchvision.models as models


def resnet50():
    model = models.resnet50(pretrained=False)

    model.conv1 = nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)

    fc_features = model.fc.in_features

    model.fc = nn.Linear(fc_features, 1)

    return model
