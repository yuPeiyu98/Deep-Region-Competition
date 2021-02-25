import cv2
import glob
import numpy as np
import os
import os.path as osp
import pandas as pd

import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from matplotlib.pyplot import imread
from PIL import Image
from skimage.color import rgb2gray, gray2rgb
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, training=True):
        super(Dataset, self).__init__()        
        self.training = training        

        self.transform = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.RandomHorizontalFlip(p=0.5 if training else 0.),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor()])

        ### parse data URL
        self.ROOT_DIR = config.ROOT_DIR
        self.IM_SIZE = config.IM_SIZE

        self.bbox_meta, self.file_meta = self.parse_url()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item    

    def parse_url(self):
        """ Returns a dictionary with image filename as 
        'key' and its bounding box coordinates as 'value' """

        data_dir = self.ROOT_DIR

        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)

        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        filenames =  [fname[:-4] for fname in filenames];
        return filename_bbox, filenames

    def load_item(self, index):
        key = self.file_meta[index]        
        bbox = self.bbox_meta[key]
        
        data_dir = self.ROOT_DIR
        
        img_path = '%s/images/%s.jpg' % (data_dir, key)
        img = self.load_imgs(img_path, bbox)

        seg_path = '%s/segmentations/%s.png' % (data_dir, key)
        seg = self.load_segs(seg_path, bbox)

        return img, seg, index
    
    def load_imgs(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)

        cimg = img.crop([x1, y1, x2, y2])
        return self.transform(cimg)        

    def load_segs(self, seg_path, bbox):
        img = Image.open(seg_path).convert('1')
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)

        cimg = img.crop([x1, y1, x2, y2])
        return self.transform_seg(cimg)        

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False
            )

            for item in sample_loader:
                yield item
