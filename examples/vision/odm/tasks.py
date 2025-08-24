# read pics from some cameras and detect the fault by resnet
from vision_detect.services.detect import CameraDetect
from vision_detect.services.model import ModelService
from vision_detect.data.device.camera import CameraReader
from vision_detect.models.output import ModelIO
from vision_detect.models.model import Model
from vision_detect.models.occlusion_detection_model import OcclusionDetectionModel
from multiprocessing import Process, Queue

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
import tkinter as tk


def visualize_occlusion_result(
        model_io: ModelIO,
        window_shape: Tuple[int, int] = (2, 2)  # 行数, 列数，比如 (2,2) 表示 2x2 排列窗口
):
    """
    Args:
        frame_result: OcclusionDetectionModelIO 实例，包含某个相机这一帧的检测结果
        window_shape: (rows, cols)，表示窗口布局，用于多相机时的排布设计
        :param window_shape:
        :param model_io:
    """
    frame_result: ModelIO.OcclusionDetectionModelIO = model_io.odm_io
    frame = cv2.cvtColor(np.array(frame_result.input), cv2.COLOR_RGB2BGR)

    cv2.putText(frame, f"Camera ID: {frame_result.ID}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    status_text = "Occluded" if frame_result.output_transformed else "Clear"
    color = (0, 0, 255) if frame_result.output_transformed else (0, 255, 0)
    cv2.putText(frame, status_text,
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 3)

    win_name = f"Camera {frame_result.ID}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    screen_w, screen_h = 1470, 956
    rows, cols = window_shape
    win_w = int(screen_w // (cols * 1.2))
    win_h = int(screen_h // (rows * 1.2))

    total_w = win_w * cols
    total_h = win_h * rows
    offset_x = (screen_w - total_w) // 2
    offset_y = (screen_h - total_h) // 2

    cam_index = int(frame_result.ID) if frame_result.ID.isdigit() else 0
    row = cam_index // cols
    col = cam_index % cols
    x = offset_x + col * win_w
    y = offset_y + row * win_h

    cv2.moveWindow(win_name, x, y)
    cv2.resizeWindow(win_name, win_w, win_h)

    cv2.imshow(win_name, frame)


def task_1():
    camera_reader = CameraReader(save_location="assets/image")
    model = OcclusionDetectionModel()
    service = ModelService(model, '/Volumes/My Passport/dataset/models/trained/best_resnet_for_faultdetect.pth', "mps")
    detect = CameraDetect(service, camera_reader)
    output = detect.detect_pic_from_cameras_serial("mps", save=True)
    for i, o in enumerate(output):
        print(output[i].odm_io)


def task_2():
    camera_reader = CameraReader(save_location="assets/image")
    model = OcclusionDetectionModel()
    service = ModelService(model, '/Volumes/My Passport/dataset/models/trained/best_resnet_for_faultdetect.pth', "mps")
    detect = CameraDetect(service, camera_reader)
    q = Queue()

    def print_func(io: ModelIO):
        print(io.odm_io)

    detect.detect_realtime_from_one_camera(q, "mps", opt=print_func, show_image=True)


# detect by 2 cameras
def task_3():
    save_path = "assets/image"
    camera_readers = [CameraReader(usb_port=0, save_location=save_path),
                      CameraReader(usb_port=1, save_location=save_path)]
    model = OcclusionDetectionModel()
    service = ModelService(model, '/Volumes/My Passport/dataset/models/trained/7_22/best.pth', "mps")
    detect = CameraDetect(service, camera_readers)
    q = Queue()

    def print_func(io: ModelIO):
        print(io.odm_io)

    detect.detect_realtime_from_cameras_serial(q, "mps", opt=print_func, show_image=True)


# detect by 1 camera
def task_4():
    save_path = "assets/image"
    camera_readers = [CameraReader(usb_port=0, save_location=save_path)]
    model = OcclusionDetectionModel()
    service = ModelService(model, '/Volumes/My Passport/dataset/models/trained/7_22/best.pth', "mps")
    detect = CameraDetect(service, camera_readers)
    q = Queue()

    def print_func(io: ModelIO):
        print(io.odm_io)

    detect.detect_realtime_from_cameras_serial(q, "mps", opt=print_func, show_image=True)


def task_5():
    save_path = "assets/image"
    camera_readers = [CameraReader(usb_port=0, save_location=save_path),
                      CameraReader(usb_port=1, save_location=save_path)]
    # camera_readers = [CameraReader(usb_port=1,save_location=save_path)]
    model = OcclusionDetectionModel()
    service = ModelService(model, '/Volumes/My Passport/dataset/models/trained/7_22/best.pth', "mps")
    detect = CameraDetect(service, camera_readers)
    q = Queue()
    detect.detect_realtime_from_cameras_serial(q, "mps", opt=visualize_occlusion_result, show_image=True)


if __name__ == "__main__":
    # task_1()
    # task_2()
    task_5()
