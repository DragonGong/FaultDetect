import cv2
from vision_detect.utils.image import ImageUtils
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
                self.camera_readers[0].capture_and_save(image, file_name)
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

    def _process_frames_from_queue(self,
                                   queue: Queue,
                                   device: Optional[str],
                                   show_image: bool,
                                   opt: Optional[Callable[[ModelIO], None]]) -> None:
        while True:
            frame = queue.get()

            if show_image:
                cv_frame = ImageUtils.pil_rgb_2_cv2_bgr(frame=frame)
                ImageUtils.show_image_plt(frame=cv_frame)
                if cv2.waitKey(100) == ord('q'):
                    break

            output = self.model_service.predict_one(device=device, image=frame)

            if opt is not None:
                opt(output)

    def detect_realtime_from_one_camera(self, queue: Queue = None, device: Optional[str] = None, fps: int = 1,
                                        get_switch: bool = True, opt: Optional[Callable[[ModelIO], None]] = None,
                                        show_image: bool = False):
        if queue is None:
            queue = Queue()
            get_switch = False
        p = Process(target=self.camera_readers[0].realtime_capture, args=(queue, ImageType.IMAGE, fps))
        p.daemon = True
        p.start()
        # for user to get data from queue
        if get_switch:
            self._process_frames_from_queue(
                queue=queue,
                device=device,
                show_image=show_image,
                opt=opt
            )

    def detect_realtime_from_cameras_serial(self, queue: Queue = None, device: Optional[str] = None, fps: int = 1,
                                            get_switch: bool = True, opt: Optional[Callable[[ModelIO], None]] = None,
                                            show_image: bool = False):
        if queue is None:
            queue = Queue()
            get_switch = False

        def realtime_read():
            while True:
                for c in self.camera_readers:
                    queue.put(c.read_image_plt())
                time.sleep(TimeUtils.seconds_per_frame(fps))

        p = Process(target=realtime_read())
        p.daemon = True
        if get_switch:
            self._process_frames_from_queue(
                queue=queue,
                device=device,
                show_image=show_image,
                opt=opt
            )
