import os.path
import random

import cv2
import numpy as np
from loguru import logger
from vision_detect.utils import constant
from PIL import Image

def add_random_occlusion(img, occlusion_ratio=0.4):
    img = np.array(img)
    h, w = img.shape[:2]
    occlusion_w = int(w * np.sqrt(occlusion_ratio))
    occlusion_h = int(h * np.sqrt(occlusion_ratio))
    x = np.random.randint(0, w - occlusion_w)
    y = np.random.randint(0, h - occlusion_h)
    img[y:y + occlusion_h, x:x + occlusion_w] = 0  # 黑色遮挡
    return Image.fromarray(img)


def add_random_occlusion_random_ratio(img, min_rat=0.2, max_rat=1):
    r = random.uniform(min_rat, max_rat)
    return add_random_occlusion(img, r)


class Fault:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def add_random_occlusion(self, img_path, occlusion_ratio=0.4):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(constant.IMAGE_NOT_EXIST + ":" + str(img_path))
            return
        img_out = add_random_occlusion(img, occlusion_ratio)
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(str(os.path.join(self.output_path, file_name)), img_out)

    def add_random_occlusion_batch(self, min_rat=0.2, max_rat=1):
        for file_name in os.listdir(self.input_path):
            occ_ratio = random.uniform(min_rat, max_rat)
            self.add_random_occlusion(str(os.path.join(self.input_path, file_name)), occ_ratio)
