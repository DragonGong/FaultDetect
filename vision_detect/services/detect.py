import torch.nn as nn
from vision_detect.models.model import Model
class Detect:
    def __init__(self,model:Model,state_path:str):
        self.model = model
        self.state_path = state_path
        pass

    def detect_pic(self):
        self.model.predict()




