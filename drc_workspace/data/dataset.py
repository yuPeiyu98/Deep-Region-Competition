import cv2
import numpy as np
import os
import os.path as osp
import pandas as pd

import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

################################################################################
### + single object datasets
################################################################################

class CubDataset(Dataset):
    def __init__(
        self, 
        config, 
        data_split=0, 
        use_flip=True
    ):
        super().__init__()        
        self.data_split = data_split
        self.use_flip = use_flip

        self.transform = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),            
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor()])
        
        self.ROOT_DIR = config.ROOT_DIR
        self.IM_SIZE = config.IM_SIZE

        self.bbox_meta, self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item    

    def collect_meta(self):
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

        splits = np.loadtxt(os.path.join(data_dir, 'train_val_test_split.txt'), int)

        for i in range(0, numImgs):            
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        filenames = [fname[:-4] for fname in filenames]

        if self.data_split == 0: # training split
            filenames = np.array(filenames)
            filenames = filenames[splits[:, 1] == 0]
            filename_bbox_ = {fname: filename_bbox[fname] for fname in filenames}
        elif self.data_split == 2: # testing split
            filenames = np.array(filenames)
            filenames = filenames[splits[:, 1] == 2]
            filename_bbox_ = {fname: filename_bbox[fname] for fname in filenames}
        elif self.data_split == -1: # all dataset
            filenames = filenames.copy()
            filename_bbox_ = filename_bbox

        print('Filtered filenames: ', len(filenames))
        return filename_bbox_, filenames

    def load_item(self, index):
        key = self.file_meta[index]        
        bbox = self.bbox_meta[key]
        
        data_dir = self.ROOT_DIR
        
        img_path = '%s/images/%s.jpg' % (data_dir, key)
        img = self.load_imgs(img_path, bbox)

        seg_path = '%s/segmentations/%s.png' % (data_dir, key)
        seg = self.load_segs(seg_path, bbox)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])
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
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item

class DogDataset(Dataset):
    def __init__(
        self, 
        config, 
        data_split=0,
        use_flip=True
    ):
        super().__init__()        
        self.data_split = data_split
        self.use_flip = use_flip

        self.transform = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor()])
        
        self.ROOT_DIR = config.ROOT_DIR
        self.IM_SIZE = config.IM_SIZE

        self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item    

    def collect_meta(self):        
        sel_indices_tr = np.load('{}/data_tr_sel.npy'.format(self.ROOT_DIR))
        sel_indices_te = np.load('{}/data_te_sel.npy'.format(self.ROOT_DIR))

        if self.data_split == 0: # training split
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr]
        elif self.data_split == 2: # testing split
            filenames = ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        elif self.data_split == -1: # all dataset
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr] \
                      + ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        return filenames

    def load_item(self, index):
        key = self.file_meta[index]        
        
        data_dir = self.ROOT_DIR            

        img_path = '%s/%s_resized.png' % (data_dir, key)
        img = self.load_imgs(img_path)

        seg_path = '%s/%s_maskresized.png' % (data_dir, key)        
        seg = self.load_segs(seg_path)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])  
        return img, seg, index
    
    def load_imgs(self, img_path):
        img = cv2.imread(img_path)
        img = Image.fromarray(img)

        return self.transform(img)        

    def load_segs(self, seg_path):
        img = Image.open(seg_path).convert('1')        

        return self.transform_seg(img)        

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item

class CarDataset(Dataset):
    def __init__(
        self, 
        config,
        data_split=0,
        use_flip=True
    ):
        super().__init__()        
        self.data_split = data_split
        self.use_flip = use_flip

        self.transform = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor()])
        
        self.ROOT_DIR = config.ROOT_DIR
        self.IM_SIZE = config.IM_SIZE

        self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item    

    def collect_meta(self):
        sel_indices_tr = np.load('{}/data_mrcnn_train_select.npy'.format(self.ROOT_DIR))
        sel_indices_te = np.load('{}/data_mrcnn_test_select.npy'.format(self.ROOT_DIR))
        
        if self.data_split == 0: # training split
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr]
        elif self.data_split == 2: # testing split
            filenames = ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        elif self.data_split == -1: # all dataset
            filenames = ['data_mrcnn/train/resized/{}'.format(token) for token in sel_indices_tr] \
                      + ['data_mrcnn/test/resized/{}'.format(token) for token in sel_indices_te]
        return filenames

    def load_item(self, index):
        key = self.file_meta[index]        
        
        data_dir = self.ROOT_DIR            

        img_path = '%s/%s_resized.png' % (data_dir, key)
        img = self.load_imgs(img_path)

        seg_path = '%s/%s_maskresized.png' % (data_dir, key)        
        seg = self.load_segs(seg_path)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])  
        return img, seg, index
    
    def load_imgs(self, img_path):
        img = cv2.imread(img_path)
        img = Image.fromarray(img)

        return self.transform(img)        

    def load_segs(self, seg_path):
        img = Image.open(seg_path).convert('1')        

        return self.transform_seg(img)        

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item

################################################################################
### + multi-object datasets
################################################################################

class ClevrDataset(Dataset):
    def __init__(
        self, 
        config, 
        data_split=0, 
        use_flip=True
    ):
        super().__init__()        
        self.data_split = data_split
        self.use_flip = use_flip

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_seg = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])
        
        self.ROOT_DIR = config.ROOT_DIR
        self.IM_SIZE = 128
        self.N_OBJ_MAX = 6 
        self.N_OBJ_MIN = 3
        self.N = 70000 

        self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta) if (self.data_split == -1) else 1000

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)
        return item    

    def collect_meta(self):                        
        filenames = []
        meta_dir = self.ROOT_DIR

        for i in range(self.N):
            meta_path = '{}/meta/{}.npz'.format(meta_dir, i)
            meta = np.load(meta_path, allow_pickle=True)

            cur_n_obj = meta['visibility'].sum()
            # filter the metas using number of visible objects
            if cur_n_obj <= self.N_OBJ_MAX and \
               cur_n_obj >= self.N_OBJ_MIN:                

                filenames.append(i)

        return filenames

    def load_item(self, index):        
        key = self.file_meta[index]
        
        data_dir = self.ROOT_DIR
        
        meta_path = '{}/meta/{}.npz'.format(data_dir, key)
        img = self.load_imgs(meta_path)        
        seg = self.load_segs(meta_path)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])
        return img, seg, index
    
    def load_imgs(self, meta_path):        
        img = np.load(meta_path)['image'][0]
        img = Image.fromarray(np.uint8(img))        

        # magical RoI borrowed from the official code
        # of IODINE
        img = img.crop([64, 29, 256, 221]) 
        return self.transform(img)

    def load_segs(self, meta_path):
        img = np.load(seg_path)['mask'][0, 0]
        img = 255 - np.dstack([img, img, img])
        img = Image.fromarray(np.uint8(img)).convert('1')

        img = img.crop([64, 29, 256, 221])        
        return self.transform_seg(img)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item

class TexturedDataset(Dataset):
    def __init__(
        self, 
        config,
        data_split=0,
        use_flip=True
    ):
        super().__init__()
        self.data_split = data_split  
        self.use_flip = use_flip

        self.transform = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_seg = transforms.Compose([
            transforms.Resize((int(config.IM_SIZE), int(config.IM_SIZE))),
            transforms.ToTensor()])

        ### parse data URL
        self.ROOT_DIR = config.ROOT_DIR
        self.IM_SIZE = config.IM_SIZE        
        self.N = 20000 if (self.data_split == -1) else 1000

        self.file_meta = self.collect_meta()

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: sample # {}'.format(index))
            item = self.load_item(0)

        return item    

    def collect_meta(self):        
        filenames = ['{:08d}'.format(i) for i in range(self.N)]
        return filenames

    def load_item(self, index):
        key = self.file_meta[index]
        
        data_dir = self.ROOT_DIR
                
        img_path = '{}/{}.png'.format(data_dir, key)
        img = self.load_imgs(img_path)

        seg_path = '{}/{}_mask.png'.format(data_dir, key)
        seg = self.load_segs(seg_path)

        if self.use_flip and np.random.uniform() > 0.5:
            img = torch.flip(img, dims=[-1])
            seg = torch.flip(seg, dims=[-1])
        return img, seg, index
    
    def load_imgs(self, img_path, bbox=None):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size                        
        return self.transform(img)

    def load_segs(self, seg_path, bbox=None):
        img = Image.open(seg_path).convert('1')
        width, height = img.size        
        return self.transform_seg(img)

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=False
            )

            for item in sample_loader:
                yield item