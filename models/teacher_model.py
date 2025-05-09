"""
The below commented code doesn't use pretrained weights and was initially trained on CIFAR-10 and Mean Test Accuracy over 45 epochs: 79.35%
"""

# import torch.nn as nn
# from torchvision.models import resnet18

# class TeacherModel(nn.Module):
#     def __init__(self, num_classes=10):
#         super(TeacherModel, self).__init__()
#         self.model = resnet18(weights=None)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights

# class TeacherModel(nn.Module):
#     def __init__(self, num_classes=10, pretrained=True):
#         super(TeacherModel, self).__init__()
#         if pretrained:
#             self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
#         else:
#             self.model = resnet18(weights=None)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TeacherModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(TeacherModel, self).__init__()
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

