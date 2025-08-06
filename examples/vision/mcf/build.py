from vision_detect.utils.Fault import add_random_occlusion_random_ratio
from vision_detect.data.build import BuildDatasets


def way_func(img):
    return add_random_occlusion_random_ratio(img, 0.2, 1)


build_datasets = BuildDatasets(
    ori_data_paths=['/Volumes/My Passport/dataset/kitti/2011_09_26-1/2011_09_26_drive_0002_extract/image_00/data',
                    '/Volumes/My Passport/dataset/kitti/2011_09_26-1/2011_09_26_drive_0002_extract/image_01/data',
                    '/Volumes/My Passport/dataset/kitti/2011_09_26-1/2011_09_26_drive_0002_extract/image_02/data',
                    '/Volumes/My Passport/dataset/kitti/2011_09_26-1/2011_09_26_drive_0002_extract/image_03/data',
                    ]
    , out_path='/Volumes/My Passport/dataset/kitti/data/test'
    , way_func=way_func, scale=0.8
)

if __name__ == '__main__':
    build_datasets.build_multi_same(output_path=[
        '/Volumes/My Passport/dataset/kitti/data3/front',
        '/Volumes/My Passport/dataset/kitti/data3/left',
        '/Volumes/My Passport/dataset/kitti/data3/right'
    ])
