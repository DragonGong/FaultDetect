import torch.cuda

from vision_detect.models.model import Model
from vision_detect.models.model import ModelIO
from PIL import Image


class ModelService:
    def __init__(self, model: Model, state_path: str, device: str = ""):
        self.model = model
        if device == "" or device is None:
            device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device_torch = torch.device(device)
        model_state = torch.load(state_path, map_location=device_torch)
        self.model.load_state_dict(model_state)
        self.model.to(device_torch)
        self.model.eval()

    def predict(self, device: str, io: ModelIO) -> ModelIO:
        input_tensor = self.model.preprocess(io, device)
        io.odm_io.output_origin = self.model(input_tensor)
        self.model.transform_output(io)
        output = io
        return output

    def predict_one(self, device: str, image: Image.Image) -> ModelIO:
        io = ModelIO()
        io.odm_io.input = image
        image_tensor = self.model.preprocess(io, device)
        output_origin = ModelIO()
        output_origin.odm_io.output_origin = self.model(image_tensor)
        self.model.transform_output(output_origin)
        output = output_origin
        return output
