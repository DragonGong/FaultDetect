from typing import Optional, Literal,get_args
from multiprocessing import Queue
import logging
import time
from typing import List
import cv2
import os
from datetime import datetime
import numpy as np
from PIL import Image
from vision_detect.utils.constant import ImageType
from vision_detect.utils.time import TimeUtils
class CameraReader:
    def __init__(self, usb_port=0, save_location='./images', image_format='.jpg'):
        self.usb_port = usb_port
        self.save_location = save_location
        self.image_format = image_format

        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)

    def _cv2pil(self, data: np.ndarray) -> Image.Image:
        data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB)
        return Image.fromarray(data)
    def _pil2cv(self,data : Image.Image)->np.ndarray:
        return np.array(data)

    def read_image_np(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.usb_port)
        if not cap.isOpened():
            logging.error(f"无法打开摄像头端口 {self.usb_port}")
            raise IOError(f"无法打开摄像头端口 {self.usb_port}")

        while True:
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logging.error(f"无法读取图像帧")
                time.sleep(1)
            else:
                break

        return frame

    def read_images_np(self, count=1) -> [np.ndarray]:
        images = []
        for _ in range(count):
            image = self.read_image_np()
            images.append(image)
        return images

    def read_image_plt(self) -> Image.Image:
        return self._cv2pil(self.read_image_np())

    def read_images_plt(self, count: int) -> List[Image.Image]:
        images = []
        for _ in range(count):
            images.append(self.read_image_plt())
        return images

    def capture_and_save(self, image=None, filename=None):
        if image is None:
            image = self.read_image_np()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}"
        if isinstance(image,Image.Image):
            image = self._pil2cv(image)
        save_path = os.path.join(self.save_location, f"{filename}{self.image_format}")

        cv2.imwrite(save_path, image)
        return save_path

    def realtime_capture(self, q: Queue, image_type: ImageType = ImageType.IMAGE,fps :int = 1):
        assert isinstance(image_type,ImageType),"Expected image_type is one of "
        t = TimeUtils.seconds_per_frame(fps)
        while True:
            q.put(self.read_image_plt())
            time.sleep(t)
