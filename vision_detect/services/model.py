from vision_detect.models.model import Model
from vision_detect.models.model import ModelIO
from PIL import Image


class ModelService:
    def __init__(self, model: Model):
        self.model = model

    def predict(self, device: str, image: Image.Image) -> ModelIO:
        image_tensor = self.model.preprocess(image,device)
        output_origin = self.model(image_tensor)
        output = self.model.transform_output(output_origin)
        return output

