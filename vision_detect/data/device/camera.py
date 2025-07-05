import logging
import time
from typing import List
import cv2
import os
from datetime import datetime
import numpy as np
from PIL import Image


class CameraReader:
    def __init__(self, usb_port=0, save_location='./images', image_format='.jpg'):
        self.usb_port = usb_port
        self.save_location = save_location
        self.image_format = image_format

        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)

    def __cv2pil(self, data: np.ndarray) -> Image.Image:
        return Image.fromarray(data)

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
        return self.__cv2pil(self.read_image_np())

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

        save_path = os.path.join(self.save_location, f"{filename}{self.image_format}")

        cv2.imwrite(save_path, image)
        return save_path
