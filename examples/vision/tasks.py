# read pics from some cameras and detect the fault by resnet
from vision_detect.services.detect import CameraDetect
from vision_detect.services.model import ModelService
from vision_detect.data.device.camera import CameraReader
from vision_detect.models.output import ModelIO
from vision_detect.models.model import Model
from vision_detect.models.occlusion_detection_model import OcclusionDetectionModel


def task_1():
    camera_reader = CameraReader()
    model = OcclusionDetectionModel()
    service = ModelService(model, '/Volumes/My Passport/dataset/models/trained/best.pth', "mps")
    detect = CameraDetect(service, camera_reader)
    output = detect.detect_pic_from_cameras_serial("mps")
    for i, o in enumerate(output):
        print(output[i].odm_io)


if __name__ == "__main__":
    task_1()
