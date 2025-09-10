from abc import ABC

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets, models
from torch.types import FileLike
from torchvision.models import ResNet18_Weights
from vision_detect.models.output import ModelIO
from vision_detect.models.model import Model


class OcclusionDetectionModel(Model):

    def __init__(self, load_pretrained: bool = False, base_model_path: FileLike = None, freeze_pretrained=False):
        super().__init__()
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

    def transform_val(self, image: Image.Image) -> torch.Tensor:
        assert isinstance(image, Image.Image), "Expected the image a Image.Image type."
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        # the image
        return transform(image)

    def preprocess(self, io: ModelIO, device: str) -> torch.Tensor:
        image: Image.Image = io.odm_io.input
        if device == "" or device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_tensor = self.transform_val(image).to(device)
        return image_tensor.unsqueeze(0)

    def transform_output(self, output: ModelIO):
        assert isinstance(output, ModelIO), "Expected output a ModelIO type."
        _, batch_max = output.odm_io.output_origin.max(1)
        output.odm_io.output_transformed = batch_max[0].item()
