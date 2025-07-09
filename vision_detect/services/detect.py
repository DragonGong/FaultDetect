import time
from vision_detect.utils.time import TimeUtils
import torch
from vision_detect.models.output import ModelIO
import torch.nn as nn
from vision_detect.data.device.camera import CameraReader
from vision_detect.services.model import ModelService
from typing import List, Union
from typing import Optional, Callable
from multiprocessing import Queue, Process
from vision_detect.utils.constant import ImageType


class CameraDetect:
    model_service: ModelService
    camera_readers: [CameraReader]

    def __init__(self, service: ModelService, cameras: Union[CameraReader, List[CameraReader]]):
        self.model_service = service
        if isinstance(cameras, list):
            self.camera_readers = cameras
        else:
            self.camera_readers = [cameras]

    def detect_pic_from_one_camera(self, device: str = "", save: bool = False, file_name: str = None) -> ModelIO:
        image = self.camera_readers[0].read_image_plt()
        if save:
            if file_name is not None:
                self.camera_readers[0].capture_and_save(image,file_name)
            else:
                self.camera_readers[0].capture_and_save(image)
        if device == "" or device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.model_service.predict_one(device=device, image=image)

    def detect_pic_from_cameras_serial(self, device: str = "", save: bool = False, file_name: str = None) -> List[
        ModelIO]:
        output = []
        if device == "" or device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i, camera in enumerate(self.camera_readers):
            image = camera.read_image_plt()
            if save:
                if file_name is not None:
                    camera.capture_and_save(image, f"{file_name}_{i}")
                else:
                    camera.capture_and_save(image)
            output.append(self.model_service.predict_one(device=device, image=image))
        return output

    def detect_realtime_from_one_camera(self, queue: Queue, device: Optional[str] = None, fps: int = 1,
                                        get_switch: bool = True, opt: Optional[Callable[[ModelIO], None]] = None):
        p = Process(target=self.camera_readers[0].realtime_capture, args=(queue, ImageType.IMAGE, fps))
        p.start()
        while get_switch:
            frame = queue.get()
            output = self.model_service.predict_one(device=device, image=frame)
            if opt is not None:
                opt(output)
