from enum import Enum
from typing import Optional, List

from PIL import Image


class IoCategory(Enum):
    Odm = 0
    Mcf = 1


class ModelIO:
    def __init__(self, category: Optional[IoCategory] = None):
        self.category: IoCategory = IoCategory.Odm
        if category is not None:
            self.category = category
        self.odm_io = self.OcclusionDetectionModelIO()
        self.mcf_io = self.MultiCameraFusionModelIO()

    class OcclusionDetectionModelIO:
        input: Image.Image
        output_origin: ...
        output_transformed: bool

        ID: str

        def __str__(self):
            return f"the result of detection is {self.output_transformed},the origin output is {self.output_origin}"

    class MultiCameraFusionModelIO:
        input: ...
        output_origin: ...
        output_transformed: List[bool]
        ids: ...

        def __init__(self):
            self.output_transformed :List[bool]= []

        def __str__(self):
            return f"the result of detection is {self.output_transformed},the origin output is {self.output_origin}"
