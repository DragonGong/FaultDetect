import torch
from vision_detect.models.output import ModelIO
import torch.nn as nn
from vision_detect.data.device.camera import CameraReader
from vision_detect.services.model import ModelService
from typing import List, Union


class CameraDetect:
    model_service: ModelService
    camera_readers: [CameraReader]

    def __init__(self, service: ModelService, cameras: Union[CameraReader, List[CameraReader]]):
        self.model_service = service
        if isinstance(cameras, list):
            self.camera_readers = cameras
        else:
            self.camera_readers = [cameras]

    def detect_pic_from_one_camera(self, device: str = "") -> ModelIO:
        image = self.camera_readers[0].read_image_plt()
        if device == "":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.model_service.predict_one(device=device, image=image)

    def detect_pic_from_cameras_serial(self, device: str = "") -> List[ModelIO]:
        output = []
        if device == "":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for camera in self.camera_readers:
            output.append(self.model_service.predict_one(device=device, image=camera.read_image_plt()))
        return output
