import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def ortho_reg(model, device):            
    cnt = 0
    loss_orth = torch.tensor(0., dtype=torch.float32, device=device)
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad \
            and len(param.shape)==4:            
            
            N = param.size(0)
            
            weight = param.view(N, -1)
            
            # (N, N)
            weight_squared = torch.matmul(
                weight, weight.transpose(1, 0))
            
            ones = torch.ones(size=weight_squared.size(), 
                              dtype=torch.float32,
                              device=device)
            
            diag = torch.eye(n=N, dtype=torch.float32, device=device)
           
            # cnt += 1
            loss_orth += ((weight_squared * (ones - diag)).square()).sum()
            
    return loss_orth

def cce(f_logits, g_logits):
    # g_logits_mu, g_logits_var = g_logits.chunk(chunks=2, dim=-1)
    g_logits_mu, __ = g_logits.chunk(chunks=2, dim=-1)
    # g_logits_var = torch.sigmoid(g_logits_var) + .25
    g_logits_var = torch.ones(1).to(f_logits.device) * .5
    g_logits_mu = torch.tanh(g_logits_mu) * 1.5
    
    l2_dis = .5 * torch.div(f_logits - g_logits_mu, 
                            g_logits_var + 1e-8).square()
    bias = torch.log(g_logits_var)
    return (l2_dis + bias).mean(dim=1).mean()

def dce(f_logits, g_logits):
    return - (F.softmax(f_logits, dim=1) \
           * F.log_softmax(g_logits, dim=1)).sum(dim=1)

def tv_loss(x):
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    return dx + dy.permute(0, 1, 3, 2)

def bias_loss(input_grid, base_grid):
    bs, h, w, __ = input_grid.size()
    base_grid = base_grid.view(1, h, w, 2).repeat(bs, 1, 1, 1).to(input_grid.device)
    return F.mse_loss(input_grid, base_grid, reduction='none')

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.padding = nn.ReflectionPad2d(sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=sobel_filter.shape, 
            padding=0)
        self.sobel_filter_horizontal.weight.data.copy_(
            torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(
            torch.from_numpy(np.array([0.0])))

        self.sobel_filter_vertical = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=sobel_filter.shape, 
            padding=0)
        self.sobel_filter_vertical.weight.data.copy_(
            torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(
            torch.from_numpy(np.array([0.0])))

    def _get_grad_mag(self, x):
        r_ch = x[:,0:1]
        g_ch = x[:,1:2]
        b_ch = x[:,2:3]        

        def _get_grad_mag_single_ch(im):
            im_pad = self.padding(im)
            grad_h = self.sobel_filter_horizontal(im_pad)
            grad_v = self.sobel_filter_vertical(im_pad)
            return torch.sqrt(
                            grad_h.square() + 
                            grad_v.square() + 1e-8
                        )

        grad_mag = (_get_grad_mag_single_ch(r_ch) + \
                    _get_grad_mag_single_ch(g_ch) + \
                    _get_grad_mag_single_ch(b_ch)) * (1. / 3.)

        def _normalize(im):
            __, __, h, w = im.size()

            return (im + F.max_pool2d(-im, kernel_size=h, stride=1)) \
                 / (F.max_pool2d(im, kernel_size=h, stride=1) 
                  + F.max_pool2d(-im, kernel_size=h, stride=1) + 1e-8)

        return _normalize(grad_mag)

    def cross_entropy(self, x, y):
        with torch.no_grad():
            Gy = self._get_grad_mag(y)            
            prob_gt = torch.cat(
                [Gy, 1. - Gy], dim=1)

        return - (prob_gt * F.log_softmax(x, dim=1)).sum(dim=1, keepdim=True)

    def forward(self, x, y):    
        return self.cross_entropy(x, y)

def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=0, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=0, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=5, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=5, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class SnakeLoss(nn.Module):
    def __init__(self):
        super(SnakeLoss, self).__init__()

    def ballon(self, mask):
        return 1 - torch.mean(mask, dim=[2, 3]).to(mask.device).mean()

    def length(self, P):
        Pf = P.roll(shifts=-1, dims=2)
        pw_dist = ((P - Pf).square().sum(dim=3) + 1e-8).sqrt()
        # return pw_dist.sum(dim=2).mean()
        return pw_dist.mean(dim=[1,2])

    def forward(self, P):
            
        loss_l = self.length(P)
                
        return loss_l

class DiceLoss(nn.Module):
    def __init__(self, _type='jaccard', smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.type = _type
        self.eps = smooth

    def forward(self, x, target):
        b = x.size(0)
        x = x.view(b, -1)
        target = target.view(b, -1)

        inse = torch.sum(x * target, dim=-1)
        if self.type == 'jaccard':
            l = torch.sum(x * x, dim=-1)
            r = torch.sum(target * target, dim=-1)
        elif self.type == 'sorensen':
            l = torch.sum(x, dim=-1)
            r = torch.sum(target, dim=-1)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + self.eps) / (l + r + self.eps)
        dice = torch.mean(dice)
        return dice

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
