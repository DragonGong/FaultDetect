import cv2
from PIL import Image
import numpy as np
from vision_detect.utils.constant import ErrorMsg


class ImageUtils:
    @staticmethod
    def show_image_plt(frame: np.ndarray):
        assert isinstance(frame, np.ndarray), ErrorMsg.get_type_error_msg(np.ndarray, type(frame))
        cv2.imshow("USB Camera", frame)

    @staticmethod
    def cv2_bgr_2_pil_rgb(frame: np.ndarray) -> Image.Image:
        assert isinstance(frame, np.ndarray), ErrorMsg.get_type_error_msg(np.ndarray, type(frame))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        return image

    @staticmethod
    def pil_rgb_2_cv2_bgr(frame: Image.Image) -> np.ndarray:
        assert isinstance(frame, Image.Image), ErrorMsg.get_type_error_msg(Image.Image, type(frame))
        if frame.mode != 'RGB':
            image = frame.convert('RGB')
        rgb_array = np.array(frame)

        if rgb_array.dtype != np.uint8:
            rgb_array = rgb_array.astype(np.uint8)
        if len(rgb_array.shape) != 3 or rgb_array.shape[2] != 3:
            raise ValueError(
                f"Input array must be 3-channel RGB (HxWx3), got shape {rgb_array.shape}"
            )

            # RGB -> BGR 转换
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
