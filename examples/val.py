import torch
from models.OcclusionDetectionModel import OcclusionDetectionModel
from PIL import Image

model_state_path = '/Volumes/My Passport/dataset/models/trained/best.pth'
model = OcclusionDetectionModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state = torch.load(model_state_path, map_location=device , weights_only= False)
state_dict = model_state.state_dict()
model.load_state_dict(state_dict)
model.eval()
image_path = '/Volumes/My Passport/dataset/kitti/data/train/00002264.png'
image = Image.open(image_path).convert("RGB")
image_tensor = model.transform_val(image).to(device)
input_tensor = image_tensor.unsqueeze(0)
if __name__ == '__main__':
    outputs = model(input_tensor)
    _, batch_max = outputs.max(1)
    print(batch_max[0].item())
