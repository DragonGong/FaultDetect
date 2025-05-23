import os.path
import random

import loguru
from PIL import Image
from tqdm import tqdm


class BuildDatasets():
    def __init__(self, ori_data_paths: [str], out_path: str, scale: float = 0.7, way_func=None
                 ):
        # scale : 正样本的比例
        self.way_func = way_func
        self.ori_data_paths = ori_data_paths
        self.scale = scale
        self.out_path = out_path

        self.ori_data = []
        for p in self.ori_data_paths:
            files = os.listdir(p)
            self.ori_data.extend([Image.open(str(os.path.join(p, f))) for f in files if not f.startswith('._')])

    def build(self, zfill_length=8):
        random.shuffle(self.ori_data)
        index = int(len(self.ori_data) * self.scale)
        list1 = self.ori_data[:index]  # 正样本
        list2 = self.ori_data[index:]  # 负样本
        if self.way_func is not None:
            list2_made = [self.way_func(img) for img in list2]
        else:
            loguru.Logger.warning("way_func is None!")
            return
        with open(str(os.path.join(self.out_path, 'labels.txt')), 'w', encoding='utf-8') as file, tqdm(
                total=len(self.ori_data)) as t:
            index = 0
            for l in list1:
                filename = str(index).zfill(zfill_length)
                l.save(os.path.join(self.out_path, filename + '.png'))
                file.write(filename + '.png'+ ',' + '0\n')
                index = index + 1
                t.update(1)
            for l in list2_made:
                filename = str(index).zfill(zfill_length)
                l.save(os.path.join(self.out_path, filename + '.png'))
                file.write(filename + '.png'+ ',' + '1\n')
                index = index + 1
                t.update(1)
