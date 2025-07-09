from abc import ABC, abstractmethod
import torch.nn as nn
from PIL import Image
from vision_detect.models.output import ModelIO
import torch


class Model(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def preprocess(self, image: Image.Image, device: str) -> torch.Tensor:
        pass

    @abstractmethod
    def transform_output(self, output: ModelIO):
        pass
