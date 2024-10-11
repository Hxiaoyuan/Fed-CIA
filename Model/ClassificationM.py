import torch.nn as nn
from model_res import *

class MyClassification(nn.Module):
    def __init__(self, num_ftrs, num_classes=10, need_pre=False):
        super(MyClassification, self).__init__()
        self.need_pre = need_pre
        if need_pre:
            self.model = ResNet50(num_classes)

        self.mlp = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.ReLU()
        )
        self.fc = nn.Linear(num_ftrs // 2, num_classes)

    def forward(self, x,  latent_output=False):
        if self.need_pre:
            x = self.model(x, latent_output=True)
        x = self.mlp(x)
        if latent_output:
            return x
        output = self.fc(x)
        return output