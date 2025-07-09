from enum import Enum

# log info
IMAGE_NOT_EXIST = 'The image file does not exist'


# const
class ImageType(Enum):
    IMAGE = "IMAGE"
    NUMPY = "NUMPY"


class ErrorMsg:
    @staticmethod
    def get_type_error_msg(expected_type, actual_type):
        return f"Expected {expected_type}, get wrong type: {actual_type}"
