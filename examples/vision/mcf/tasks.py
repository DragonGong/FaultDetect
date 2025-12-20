from vision_detect.models.multi_camera_fusion import MultiCameraFusion
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

from queue import Queue as tQueue
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
                (10, frame.shape[0] - 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 3)

    win_name = f"Camera {int(frame_result.ID) - 1}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    screen_w, screen_h = 1470, 956
    rows, cols = window_shape
    win_w = int(screen_w // (cols * 1.2))
    win_h = int(screen_h // (rows * 1.2))

    total_w = win_w * cols
    total_h = win_h * rows
    offset_x = (screen_w - total_w) // 2
    offset_y = (screen_h - total_h) // 2

    cam_index = int(frame_result.ID)  if frame_result.ID.isdigit() else 0
    row = cam_index // cols
    col = cam_index % cols
    x = offset_x + col * win_w
    y = offset_y + row * win_h

    cv2.moveWindow(win_name, x, y)
    cv2.resizeWindow(win_name, win_w, win_h)

    cv2.imshow(win_name, frame)
    cv2.waitKey(1)


def visualize_mcf_result(
        model_io: 'ModelIO',
        window_shape: Tuple[int, int] = (2, 2)  # 行数, 列数，比如 (2,2) 表示 2x2 排列窗口
):
    """
    可视化多相机融合（MCF）的遮挡检测结果。

    Args:
        model_io: 包含 mcf_io 的 ModelIO 实例
        window_shape: (rows, cols)，窗口布局，用于多相机排布
    """
    mcf_result = model_io.mcf_io  # MultiCameraFusionModelIO 实例
    if not hasattr(mcf_result, 'input') or not isinstance(mcf_result.input, (list, tuple)):
        raise ValueError("mcf_io.input 应为包含各相机图像的列表")

    num_cameras = len(mcf_result.input)
    if len(mcf_result.output_transformed) != num_cameras:
        raise ValueError("output_transformed 长度应与输入图像数量一致")
    if len(mcf_result.ids) != num_cameras:
        raise ValueError("ids 长度应与输入图像数量一致")

    screen_w, screen_h = 1470, 956
    rows, cols = window_shape
    win_w = int(screen_w // (cols * 1.2))
    win_h = int(screen_h // (rows * 1.2))

    total_w = win_w * cols
    total_h = win_h * rows
    offset_x = (screen_w - total_w) // 2
    offset_y = (screen_h - total_h) // 2

    for idx in range(num_cameras):
        cam_id = mcf_result.ids[idx]
        frame_pil = mcf_result.input[idx]
        is_occluded = mcf_result.output_transformed[idx]

        # 转为 OpenCV BGR 格式
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 绘制相机 ID
        cv2.putText(frame, f"Camera ID: {cam_id}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        # 绘制遮挡状态
        status_text = "Occluded" if is_occluded else "Clear"
        color = (0, 0, 255) if is_occluded else (0, 255, 0)
        cv2.putText(frame, status_text,
                    (10, frame.shape[0] - 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 3)

        # 窗口命名（兼容原逻辑：ID 从 0 开始）
        win_name = f"Camera {int(cam_id) - 1}" if cam_id.isdigit() else f"Camera_{cam_id}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # 计算窗口位置（按 ID 或按 idx 排列？这里建议按 idx 更稳定）
        # 若 cam_id 是连续数字且从1开始，可用 cam_id；否则用 idx
        try:
            cam_index = int(cam_id) if cam_id.isdigit() else idx
        except:
            cam_index = idx

        row = cam_index // cols
        col = cam_index % cols
        x = offset_x + col * win_w
        y = offset_y + row * win_h

        cv2.moveWindow(win_name, x, y)
        cv2.resizeWindow(win_name, win_w, win_h)
        cv2.imshow(win_name, frame)

    cv2.waitKey(1)  # 注意：只调用一次 waitKey，避免阻塞



def task_5_V1():
    save_path = "assets/image"
    # camera_readers = [CameraReader(usb_port=1, save_location=save_path),
    #                   CameraReader(usb_port=2, save_location=save_path),
    #                   CameraReader(usb_port=3,save_location=save_path)]
    camera_readers = [CameraReader(usb_port=0,save_location=save_path),
                      ]
    model = MultiCameraFusion()
    service = ModelService(model, r"assets/mcf_model/best.pth", "cpu")
    print("model is loaded")
    with CameraDetect(service,camera_readers) as detect:
        q = Queue()
        detect.detect_realtime_from_cameras_parallel(q, "cpu", opt=visualize_mcf_result, show_image=False,fps=10)


if __name__ == "__main__":
    task_5_V1()
