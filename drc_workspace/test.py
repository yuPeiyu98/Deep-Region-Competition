import argparse
import cv2
import numpy as np
import os
import os.path as osp
import random
import torch
import torch.nn as nn

from data.dataset import *

from PIL import Image
from shutil import copyfile
from src.utils import Config, Progbar, create_dir
from src.utils import stitch_images, imsave

from src.models import EABPModel
from torch.utils.data import DataLoader

def run(model, config):
    # dataset configuration
    def get_dataset(config):
        if config.DATA.lower() == "cub":            
            val_dataset = CubDataset(
                                config, 
                                data_split=config.TEST_SPLIT,
                                use_flip=False
                            )            
        elif config.DATA.lower() == "dog":            
            val_dataset = DogDataset(
                                config, 
                                data_split=config.TEST_SPLIT,
                                use_flip=False
                            )            
        elif config.DATA.lower() == "car":            
            val_dataset = CarDataset(
                                config, 
                                data_split=config.TEST_SPLIT, 
                                use_flip=False
                            )            
        elif config.DATA.lower() == "clevr":            
            val_dataset = ClevrDataset(
                                config, 
                                data_split=config.TEST_SPLIT, 
                                use_flip=False
                            )            
        elif config.DATA.lower() == "textured":            
            val_dataset = TexturedDataset(
                                config, 
                                data_split=config.TEST_SPLIT, 
                                use_flip=False
                            )            
        else:
            raise ValueError("Unknown dataset.")
        return val_dataset
    
    val_dataset = get_dataset(config)

    token = '1000_iter300_viz'

    samples_path = os.path.join(config.PATH, 'samples_test_{}'.format(token))    
    gen_path = os.path.join(config.PATH, 'generated_test_{}'.format(token))        

    log_file = osp.join(
        config.PATH, 'log_' + model.model_name + '_test_{}.dat'.format(token))

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

    def save_gen(img, n, token):
        for i, im in enumerate(img):
            im = (im.permute(1, 2, 0) + 1) / 2 * 255.0
            im = Image.fromarray(
                im.cpu().numpy().astype(np.uint8).squeeze())

            path = osp.join(gen_path, model.model_name, token)
            name = osp.join(
                path, str(n + i).zfill(5) + ".png")
            create_dir(path)            
            im.save(name)    

    def viz(iteration, im_t, seg_t, im_p, fg_wp, bg_wp, pi_f, fg, bg):                
        image_per_row = 2
        if config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            postprocess((im_t + 1) / 2.),
            postprocess((im_p + 1) / 2.),
            postprocess((fg_wp + 1) / 2.),
            postprocess((bg_wp + 1) / 2.),
            postprocess((bg + 1) / 2.),
            postprocess(pi_f),
            postprocess(seg_t),                    
            img_per_row = image_per_row
        )

        path = osp.join(samples_path, model.model_name)
        name = osp.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def test():
        test_loader = DataLoader(
                dataset=val_dataset,
                batch_size=config.BATCH_SIZE,
                num_workers=2,
                drop_last=False,
                shuffle=False
            )

        total = len(val_dataset)

        if total == 0:
            print('No data was provided!'\
                  +' Check value in the configuration file.')
            return

        progbar = Progbar(total, width=20, 
                        stateful_metrics=['epoch', 'iter'])

        epoch = 0
        iou_s = 0
        dice_s = 0
        cnt = 0
        
        for iteration, items in enumerate(test_loader):            
            model.eval()

            im_t, seg_t, index = cuda(*items)                
            # inference                
            zp = model.sample_langevin_posterior(im_t)            
            with torch.no_grad():
                # learn                
                im_p, fg, bg_wp, pi_f, pi_b, bg, __, __, __ = model(zp, im_t)
                
                bs = im_p.size(0)

                ### calculate metric
                pred = pi_f > 0.5
                gt = seg_t > 0.5

                iou = (pred * gt).view(bs, -1).sum(dim=-1) / \
                      ((pred + gt) > 0).view(bs, -1).sum(dim=-1)
                dice = 2 * (pred * gt).view(bs, -1).sum(dim=-1) / \
                       (pred.view(bs, -1).sum(dim=-1) + gt.view(bs, -1).sum(dim=-1))                

                iou_s += iou.sum().item()
                dice_s += dice.sum().item()

            logs = [
                ("epoch", epoch),
                ("iter", iteration),
                ("IoU", iou.mean().item()),
                ("Dice", dice.mean().item())
            ]

            progbar.add(len(im_t), values=logs)

            # save generated sample
            save_gen(im_p, cnt, 'gen')
            save_gen(im_t, cnt, 'gt')
            cnt += bs

            # log model at checkpoints
            if config.LOG_INTERVAL and \
                iteration % config.LOG_INTERVAL == 0:
                log(logs)                                

            # sample model at checkpoints
            if config.SAMPLE_INTERVAL and \
                iteration % config.SAMPLE_INTERVAL == 0:
                viz(iteration, im_t, seg_t, im_p, 
                       fg, bg_wp, pi_f, fg, bg)        
        
        print('Avg Iou: {}, Dice: {}'.format(
            iou_s / cnt, dice_s / cnt))
        print('\nEnd testing....')

    test()

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
    cv2.setNumThreads(0)

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
