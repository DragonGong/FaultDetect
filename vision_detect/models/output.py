class ModelIO:
    def __init__(self):
        self.odm_io = self.OcclusionDetectionModelIO()

    class OcclusionDetectionModelIO:
        input: ...
        output_origin:...
        output_transformed: bool
