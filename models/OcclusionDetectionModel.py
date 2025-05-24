import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.types import FileLike
from torchvision.models import ResNet18_Weights

class OcclusionDetectionModel(nn.Module):
    def __init__(self, load_pretrained: bool = False, base_model_path: FileLike = None, freeze_pretrained=False):
        super(OcclusionDetectionModel, self).__init__()
        if load_pretrained:
            if base_model_path is None:
                self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用预训练的ResNet-18模型
            else:
                self.base_model = models.resnet18(weights=None)
                state_dict = torch.load(base_model_path)
                self.base_model.load_state_dict(state_dict)
        else:
            self.base_model = models.resnet18(weights=None)

        if freeze_pretrained:
            params = self.base_model.parameters()
            for p in params:
                p.requires_grad = False

        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # 修改最后一层，输出2个类别（遮挡与未遮挡）

    def forward(self, x):
        return self.base_model(x)

    def transform_val(self, image):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        return transform(image)
