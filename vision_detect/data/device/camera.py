import logging

import cv2
import os
from datetime import datetime
import numpy as np


class CameraReader:
    def __init__(self, usb_port=0, save_location='./images', image_format='.jpg'):
        self.usb_port = usb_port
        self.save_location = save_location
        self.image_format = image_format

        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)

    def read_image(self):
        cap = cv2.VideoCapture(self.usb_port)
        if not cap.isOpened():
            logging.error(f"无法打开摄像头端口 {self.usb_port}")
            raise IOError(f"无法打开摄像头端口 {self.usb_port}")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            logging.error(f"无法读取图像帧")
            raise RuntimeError("无法读取图像帧")

        return frame

    def read_images(self, count=1):
        images = []
        for _ in range(count):
            image = self.read_image()
            images.append(image)
        return images

    def capture_and_save(self, image=None, filename=None):
        if image is None:
            image = self.read_image()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}"

        save_path = os.path.join(self.save_location, f"{filename}{self.image_format}")

        cv2.imwrite(save_path, image)
        return save_path
