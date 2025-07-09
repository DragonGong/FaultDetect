# read pics from some cameras and detect the fault by resnet
from vision_detect.services.detect import CameraDetect
from vision_detect.services.model import ModelService
from vision_detect.data.device.camera import CameraReader
from vision_detect.models.output import ModelIO
from vision_detect.models.model import Model
from vision_detect.models.occlusion_detection_model import OcclusionDetectionModel
from multiprocessing import Process, Queue


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


def task_3():
    save_path = "assets/image"
    camera_readers = [CameraReader(usb_port=0, save_location=save_path),
                      CameraReader(usb_port=1, save_location=save_path)]
    model = OcclusionDetectionModel()
    service = ModelService(model, '/Volumes/My Passport/dataset/models/trained/best_resnet_for_faultdetect.pth', "mps")
    detect = CameraDetect(service, camera_readers)
    q = Queue()

    def print_func(io: ModelIO):
        print(io.odm_io)

    detect.detect_realtime_from_cameras_serial(q, "mps", opt=print_func, show_image=True)


if __name__ == "__main__":
    # task_1()
    # task_2()
    task_3()