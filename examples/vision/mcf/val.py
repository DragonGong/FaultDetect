from vision_detect.models.output import ModelIO
from PIL import Image
from vision_detect.models.model import Model
from vision_detect.models.multi_camera_fusion import MultiCameraFusion
import torch

model = MultiCameraFusion()
device = torch.device('cpu')
model.load_state_dict(torch.load('/Volumes/My Passport/dataset/models/trained/8_5/best.pth', map_location=device))
model.to(device)
model.eval()

image_path = [
    '/Users/mac/data/workspace/eastwind/FaultDetect/assets/image/image_20250709_154603.jpg',
    '/Volumes/My Passport/dataset/kitti/data2/test/000000000.png',
    '/Volumes/My Passport/dataset/kitti/data2/test/000000118.png',
]
image_list = []
for path in image_path:
    image_list.append(model.preprocess(Image.open(path).convert('RGB'), device='cpu'))
image_tensor = torch.stack(image_list)
image_input = image_tensor.unsqueeze(0)
if __name__ == "__main__":
    output = model(image_input)
    io = ModelIO()
    io.mcf_io.input = image_input
    io.mcf_io.output_origin = output
    model.transform_output(io)
    print(io.mcf_io)
