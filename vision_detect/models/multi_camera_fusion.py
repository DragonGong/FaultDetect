import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.types import FileLike
from torchvision.models import ResNet18_Weights
from .model import Model
from vision_detect.models.output import ModelIO
from torchvision import transforms


class CameraCNN(nn.Module):
    def __init__(self, output_dim=512, load_pretrained: bool = False, base_model_path: FileLike = None,
                 freeze_pretrained=False):
        super().__init__()
        if load_pretrained:
            if base_model_path is None:
                resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用预训练的ResNet-18模型
            else:
                resnet = models.resnet18(weights=None)
                state_dict = torch.load(base_model_path)
                resnet.load_state_dict(state_dict)
        else:
            resnet = models.resnet18(weights=None)

        if freeze_pretrained:
            params = resnet.parameters()
            for p in params:
                p.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的fc
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_proj = nn.Linear(512, output_dim)

    def forward(self, x):  # x: [B, 3, H, W]
        feat = self.backbone(x)  # [B, 512, H', W']
        feat = self.pool(feat).view(x.size(0), -1)  # [B, 512]
        return self.out_proj(feat)  # [B, output_dim]


class MultiCameraFusion(Model):

    def __init__(self, cam_num=3, embed_dim=512, nhead=4, num_classes=2, load_pretrained: bool = False,
                 base_model_path: FileLike = None,
                 freeze_pretrained=False):
        super().__init__()
        self.cnn = CameraCNN(output_dim=embed_dim, load_pretrained=load_pretrained,
                             base_model_path=base_model_path, freeze_pretrained=freeze_pretrained)
        self.pos_emb = nn.Parameter(torch.randn(cam_num, embed_dim))
        self.attn = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=2048,
            activation='gelu',
            batch_first=False
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.cnn(x)  # [B*N, 512]
        feats = feats.view(B, N, -1).permute(1, 0, 2)  # [N, B, 512]
        pos = self.pos_emb[:N].unsqueeze(1).expand(-1, B, -1)
        feats = feats + pos
        fused = self.attn(feats)  # [N, B, 512]
        fused = fused.permute(1, 0, 2)  # [B, N, 512]

        logits = self.classifier(fused)  # [B, N, num_classes]
        return logits

    @staticmethod
    def transform_func():
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def transform_val(self, image: Image.Image) -> torch.Tensor:
        assert isinstance(image, Image.Image), "Expected the image a Image.Image type."
        transform = self.transform_func()
        # the image
        return transform(image)

    def preprocess(self, image: Image.Image, device: str) -> torch.Tensor:
        if device == "" or device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        t = self.transform_val(image).to(device)
        return t

    def transform_output(self, output: ModelIO):
        _, batch_max_list = output.mcf_io.output_origin.max(2)
        batch_max_list = batch_max_list[0]
        for value in batch_max_list:
            output.mcf_io.output_transformed.append(value.item())
