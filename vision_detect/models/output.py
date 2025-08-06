class ModelIO:
    def __init__(self):
        self.odm_io = self.OcclusionDetectionModelIO()
        self.mcf_io = self.MultiCameraFusionModelIO()

    class OcclusionDetectionModelIO:
        input: ...
        output_origin: ...
        output_transformed: bool

        def __str__(self):
            return f"the result of detection is {self.output_transformed},the origin output is {self.output_origin}"

    class MultiCameraFusionModelIO:
        input: ...
        output_origin: ...
        output_transformed: [bool]

        def __init__(self):
            self.output_transformed = [bool]

        def __str__(self):
            return f"the result of detection is {self.output_transformed},the origin output is {self.output_origin}"
