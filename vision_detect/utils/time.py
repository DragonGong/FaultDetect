class TimeUtils:
    @staticmethod
    def seconds_per_frame(fps: float) -> float:
        if fps <= 0:
            raise ValueError("FPS must be greater than 0.")
        return 1.0 / fps