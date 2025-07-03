import torch
from vision_detect.models.occlusion_detection_model import OcclusionDetectionModel
from PIL import Image

model_state_path = '/Volumes/My Passport/dataset/models/trained/best.pth'
model = OcclusionDetectionModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state = torch.load(model_state_path, map_location=device , weights_only= False)
state_dict = model_state.state_dict()
model.load_state_dict(state_dict)
model.eval()
image_path = '/Volumes/My Passport/dataset/kitti/data/real/1931748082808_.pic.jpg'
image = Image.open(image_path).convert("RGB")
image_tensor = model.transform_val(image).to(device)
input_tensor = image_tensor.unsqueeze(0)
if __name__ == '__main__':
    outputs = model(input_tensor)
    _, batch_max = outputs.max(1)
    print(batch_max[0].item())  # 0 未遮挡  1 遮挡
