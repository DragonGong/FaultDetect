import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models


class OcclusionDetectionModel(nn.Module):
    def __init__(self, base_model_path=None, freeze_pretrained=False):
        super(OcclusionDetectionModel, self).__init__()
        if base_model_path is None:
            self.base_model = models.resnet18(pretrained=True)  # 使用预训练的ResNet-18模型
        else:
            state_dict = torch.load(base_model_path)
            self.base_model = models.resnet18(pretrained=False)
            self.base_model.load_state_dict(state_dict)

        if freeze_pretrained:
            params = self.base_model.parameters()
            for p in params:
                p.requires_grad = False

        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # 修改最后一层，输出2个类别（遮挡与未遮挡）

    def forward(self, x):
        return self.base_model(x)
