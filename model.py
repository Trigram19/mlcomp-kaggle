import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from catalyst.contrib import registry

@registry.Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v2(pretrained=True)
        self.model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(in_features=1280, out_features=14, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x
