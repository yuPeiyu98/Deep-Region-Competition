import argparse
import cv2
import numpy as np
import os
import os.path as osp
import random
import torch
import torch.nn as nn

from data.dataset import *

from shutil import copyfile
from src.utils import Config, Progbar, create_dir
from src.utils import stitch_images, imsave

from src.models import EABPModel
from torch.utils.data import DataLoader

def run(model, config):
    # dataset configuration
    def get_dataset(config):
        if config.DATA.lower() == "cub":
            train_dataset = CubDataset(
                                config, 
                                data_split=config.TRAIN_SPLIT,
                                use_flip=True
                            )
            val_dataset = CubDataset(
                                config, 
                                data_split=config.VAL_SPLIT,
                                use_flip=False
                            )            
        elif config.DATA.lower() == "dog":
            train_dataset = DogDataset(
                                config, 
                                data_split=config.TRAIN_SPLIT, 
                                use_flip=True
                            )
            val_dataset = DogDataset(
                                config, 
                                data_split=config.VAL_SPLIT,
                                use_flip=False
                            )            
        elif config.DATA.lower() == "car":
            train_dataset = CarDataset(
                                config, 
                                data_split=config.TRAIN_SPLIT, 
                                use_flip=True
                            )
            val_dataset = CarDataset(
                                config, 
                                data_split=config.VAL_SPLIT,
                                use_flip=False
                            )            
        elif config.DATA.lower() == "clevr":
            train_dataset = ClevrDataset(
                                config, 
                                data_split=config.TRAIN_SPLIT, 
                                use_flip=True
                            )
            val_dataset = ClevrDataset(
                                config, 
                                data_split=config.VAL_SPLIT,
                                use_flip=False
                            )            
        elif config.DATA.lower() == "textured":
            train_dataset = TexturedDataset(
                                config, 
                                data_split=config.TRAIN_SPLIT,
                                use_flip=True
                            )
            val_dataset = TexturedDataset(
                                config, 
                                data_split=config.VAL_SPLIT,
                                use_flip=False
                            )
        else:
            raise ValueError("Unknown dataset.")
        return train_dataset, val_dataset

    train_dataset, val_dataset = get_dataset(config)
    sample_iterator = val_dataset.create_iterator(
        config.SAMPLE_SIZE)

    samples_path = os.path.join(config.PATH, 'samples')        

    log_file = osp.join(
        config.PATH, 'log_' + model.model_name + '.dat')

    def log(logs):
        with open(log_file, 'a') as f:
            f.write('%s\n' % ' '.join(
                [str(item[1]) for item in logs]))

    def cuda(*args):
        return (item.to(config.DEVICE) for item in args)

    def postprocess(img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def normalize(img):
        if img.max() - img.min() < 1e-5:
            if img.max() < 1e-5:
                img = torch.zeros(img.shape)
            else:
                img = torch.ones(img.shape)
        else:
            img = (img - img.min()) / (img.max() - img.min())
        return img

    def sample():
        # do not sample when validation set is empty
        if len(val_dataset) == 0:
            return

        model.eval()
        
        items = next(sample_iterator)
        im_t, seg_t, __ = cuda(*items)

        iteration = model.iteration
        # inference
        z = model.sample_langevin_posterior(im_t)
        with torch.no_grad():
            # forward
            im_p, fg, bg_wp, pi_f, pi_b, bg, __, __, __ = model(z, im_t)            

        image_per_row = 2
        if config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            postprocess((im_t + 1) / 2.),
            postprocess((im_p + 1) / 2.),
            postprocess((fg + 1) / 2.),
            postprocess((bg_wp + 1) / 2.),
            postprocess(pi_f),
            postprocess((bg + 1) / 2.),
            img_per_row = image_per_row
        )

        path = osp.join(samples_path, model.model_name)
        name = osp.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def train():
        train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=config.BATCH_SIZE,
                num_workers=4,
                drop_last=True,
                shuffle=True
            )

        epoch = 0
        keep_training = True
        max_iteration = int(float((config.MAX_ITERS)))
        total = len(train_dataset)

        if total == 0:
            print('No training data was provided!'\
                  +' Check value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, 
                            stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                model.train()

                im_t, __, index = cuda(*items)

                # learn                
                logs = model.learn(im_t, index)
                
                iteration = model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(im_t), values=logs)

                # log model at checkpoints
                if config.LOG_INTERVAL and \
                    iteration % config.LOG_INTERVAL == 0:
                    log(logs)

                # sample model at checkpoints
                if config.SAMPLE_INTERVAL and \
                    iteration % config.SAMPLE_INTERVAL == 0:
                    sample()                

                # save model at checkpoints
                if config.SAVE_INTERVAL and \
                    iteration % config.SAVE_INTERVAL == 0:
                    model.save()

        print('\nEnd training....')
    train()

def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, 
        reads from config file if not specified
    """

    config = load_config(mode)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = \
                    ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        # cudnn auto-tuner
        # torch.backends.cudnn.benchmark = True   
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 
    # (prevents deadlocks with pytorch dataloader)
    # cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = EABPModel(config).to(config.DEVICE)
    model.load()

    # model training    
    config.print()
    print('\nstart training...\n')

    with torch.autograd.set_detect_anomaly(True):
        run(model, config)

def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, 
        reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', 
        '--checkpoints', 
        type=str, 
        default='./checkpoints', 
        help='model checkpoints path (default: ./checkpoints)')        

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)    

    return config


if __name__ == "__main__":
    main()
