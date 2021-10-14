import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

###################################################################
####################### BASIC ARCH COMPONENTS #####################
###################################################################

class GLU(nn.Module):
    """ GLU activation halves the channel number once applied """
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class UpBlock(nn.Module):
    """ upsample the feature map by a factor of 2x """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.block = nn.Sequential(
              nn.Upsample(scale_factor=2, mode='bilinear'),
              nn.Conv2d(in_channels, 
                        out_channels, 
                        kernel_size=3, 
                        stride=1,
                        padding=1, 
                        bias=False),
              nn.InstanceNorm2d(out_channels,
                                affine=True, 
                                track_running_stats=False),
              nn.LeakyReLU(inplace=True)
          )
    
    def forward(self, x):
        return self.block(x)

class SameBlock(nn.Module):
    """ shape-preserving feature transformation """
    def __init__(self, in_channels, out_channels, r=.01):
        super(SameBlock, self).__init__()
        
        self.block = nn.Sequential(
              nn.Conv2d(in_channels, 
                        out_channels, 
                        kernel_size=3, 
                        stride=1,
                        padding=1, 
                        bias=False),
              nn.InstanceNorm2d(out_channels,
                                affine=True, 
                                track_running_stats=False),
              nn.LeakyReLU(r, inplace=True)
          )
    
    def forward(self, x):
        return self.block(x)

class DownBlock(nn.Module):
    """ down-sample the feature map by a factor of 2x """
    def __init__(self, in_channels, out_channels, use_spc_norm=False):
        super(DownBlock, self).__init__()
        
        self.block = nn.Sequential(
              spectral_norm(
                nn.Conv2d(in_channels, 
                        out_channels, 
                        kernel_size=4, 
                        stride=2,
                        padding=1, 
                        bias=False), 
                mode=use_spc_norm),
              nn.InstanceNorm2d(out_channels,
                                affine=True, 
                                track_running_stats=False),
              nn.LeakyReLU(0.2, inplace=True)
          )
    
    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            SameBlock(in_channels, in_channels),            
            nn.Conv2d(in_channels, 
                      in_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=1, 
                      bias=False),
            nn.InstanceNorm2d(in_channels,
                              affine=True, 
                              track_running_stats=False),
        )


    def forward(self, x):
        return self.block(x) + x

class BaseNetwork(nn.Module):        
    def __init__(self, name):
        super(BaseNetwork, self).__init__()
        self.name = name        

    def init_weights(self, init_type='orthogonal', gain=1.):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/
        9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
                or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class BaseDecoder(nn.Module):
    def __init__(self, block_chns=[128, 1024, 512, 256, 128, 64]):
        super(BaseDecoder, self).__init__()
        blocks = []        
        for i in range(0, len(block_chns)-1):
            block = UpBlock(in_channels=block_chns[i], 
                            out_channels=block_chns[i+1])
            blocks.append(block)        

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class BaseEncoder(nn.Module):
    def __init__(self, in_channels, 
                       out_channels, 
                       block_chns=[64, 128, 256, 512, 1024],
                       use_spc_norm=False):
        super(BaseEncoder, self).__init__()
        blocks = [] 
        blocks.append(DownBlock(in_channels=in_channels,
                                out_channels=block_chns[0],
                                use_spc_norm=use_spc_norm))
        for i in range(0, len(block_chns)-1):
            block = DownBlock(in_channels=block_chns[i], 
                              out_channels=block_chns[i+1],
                              use_spc_norm=use_spc_norm)
            blocks.append(block)        
        blocks.append(spectral_norm(
                          nn.Conv2d(in_channels=block_chns[-1],
                                    out_channels=out_channels,
                                    kernel_size=6,
                                    stride=4,
                                    padding=1,
                                    bias=True),
                          mode=use_spc_norm
                          ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class ImageHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImageHead, self).__init__()        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=1, 
                      bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class MaskHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskHead, self).__init__()        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=1, 
                      bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

###################################################################
####################### GENERATOR NETWORKS ########################
###################################################################

class FgNet(BaseNetwork):
    def __init__(self, name='fg_net',
                       z_dim=64, 
                       block_chns=[128, 
                                   1024, 
                                   512, 
                                   256, 
                                   128, 
                                   64],
                       kf=200,
                       init_weights=True):
        super(FgNet, self).__init__(name=name)

        ############### generator arch ##############
        self.fc = nn.Sequential(
              nn.Linear(z_dim, block_chns[0] * 4 * 4, bias=True),
              nn.LeakyReLU(0.2, inplace=True)
            )

        # object feature base
        # z -> (B, 64, 128, 128)
        self.decode_base = BaseDecoder(block_chns=block_chns)

        self.im_head = ImageHead(in_channels=64, out_channels=3)
        self.ma_head = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=1, 
                      kernel_size=3, 
                      stride=1,
                      padding=1, 
                      bias=True))

        ############### encoder arch ##############
        self.encode_base = BaseEncoder(in_channels=3, 
                                       out_channels=kf,
                                       use_spc_norm=False)
        if init_weights:
            self.init_weights()        

    def forward(self, z):
        bs = z.size(0)

        f_lat = self.fc(z).view(bs, -1, 4, 4)
        obj_feat = self.decode_base(f_lat)

        ### image 
        app = self.im_head(obj_feat)

        ### mask
        ma = self.ma_head(obj_feat)

        ### logits for MI
        app_and_ma = app * ma.sigmoid()
        f_logits = self.encode_base(app_and_ma)
        f_logits = f_logits.view(bs, -1)
        
        return app, ma, f_logits, f_lat

class BgNet(BaseNetwork):
    def __init__(self, name='bg_net',
                       z_dim=32,
                       block_chns=[128, 
                                   1024, 
                                   512, 
                                   256, 
                                   128, 
                                   64],
                       kb=200,
                       init_weights=True):
        super(BgNet, self).__init__(name=name)

        ############### generator arch ##############
        self.fc = nn.Sequential(
              nn.Linear(z_dim, block_chns[0] * 4 * 4, bias=True),
              nn.LeakyReLU(0.2, inplace=True)
            )

        # object feature base
        # z -> (B, 64, 128, 128)
        self.decode_base = BaseDecoder(block_chns=block_chns)        

        # final decoding layers
        self.im_head = ImageHead(in_channels=64, out_channels=3)
        self.ma_head = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=1, 
                      kernel_size=3, 
                      stride=1,
                      padding=1, 
                      bias=True))

        ############### encoder arch ##############
        self.encode_base = BaseEncoder(in_channels=3, 
                                       out_channels=kb,
                                       use_spc_norm=False)
        if init_weights:
            self.init_weights()

    def forward(self, z):
        bs = z.size(0)

        b_lat = self.fc(z).view(bs, -1, 4, 4)            
        bg_feat = self.decode_base(b_lat)

        ### image         
        bg = self.im_head(bg_feat)
        ### mask 
        ma = self.ma_head(bg_feat)

        ### logits for MI        
        b_logits = self.encode_base(bg * ma.sigmoid())
        b_logits = b_logits.view(bs, -1)
        return bg, ma, b_logits, b_lat

class SpNet(BaseNetwork):
    def __init__(self, name='sp_net',
                       z_dim=128, 
                       zf_dim=128,
                       zb_dim=128,
                       block_chns=[128, 
                                   1024, 
                                   512, 
                                   256, 
                                   128, 
                                   64],
                       init_weights=True):
        super(SpNet, self).__init__(name=name)

        ############### generator arch ##############   
        self.fc = nn.Sequential(
              nn.Linear(z_dim, block_chns[0] * 4 * 4, bias=True),
              nn.LeakyReLU(0.2, inplace=True)
            )

        # feature base
        # z -> (B, 64, 128, 128)
        block_chns[0] += zf_dim
        self.decode_base_deform = BaseDecoder(block_chns=block_chns)
        # deform grid
        self.decode_deform = ImageHead(64, 2)        

        ############### encoder arch ##############
        self.encode_base = BaseEncoder(in_channels=4, 
                                       out_channels=z_dim * 2)

        if init_weights:
            self.init_weights()

    def forward(self, z, b_lat):
        bs = z.size(0)

        sp_lat = self.fc(z)
 
        ############### bg deform estimation ##############
        deform_lat = sp_lat.view(bs, -1, 4, 4)
        deform_lat = torch.cat([
                        b_lat, 
                        deform_lat
                     ], dim=1)
        deform_lat = self.decode_base_deform(deform_lat)
        deform_grid = self.decode_deform(deform_lat)
        # (B, 2, H, W) -> (B, H, W, 2)
        deform_grid = deform_grid.permute(0, 2, 3, 1)        

        return deform_grid

###################################################################
########################## LATENT EBMS ############################
###################################################################

class CEBMNet(BaseNetwork):
    def __init__(self, name='ebm_net',
                       zf_dim=64,
                       zb_dim=32,
                       zsp_dim=128,
                       nef=200,
                       Kf=200,
                       Kb=200,
                       use_spc_norm=False,
                       init_weights=True):
        super(CEBMNet, self).__init__(name=name)
        self.zf_dim = zf_dim
        self.zb_dim = zb_dim
        self.zsp_dim = zsp_dim

        self.fg_model = nn.Sequential(
              spectral_norm(
                nn.Linear(zf_dim, nef),
                mode=use_spc_norm
              ),
              nn.LeakyReLU(0.2, inplace=True),
              spectral_norm(
                nn.Linear(nef, nef),
                mode=use_spc_norm
              ),
              nn.LeakyReLU(0.2, inplace=True),
              spectral_norm(
                nn.Linear(nef, Kf),
                mode=use_spc_norm
              )
            )
        self.bg_model = nn.Sequential(
              spectral_norm(
                nn.Linear(zb_dim, nef),
                mode=use_spc_norm
              ),
              nn.LeakyReLU(0.2, inplace=True),
              spectral_norm(
                nn.Linear(nef, nef),
                mode=use_spc_norm
              ),
              nn.LeakyReLU(0.2, inplace=True),
              spectral_norm(
                nn.Linear(nef, Kb),
                mode=use_spc_norm
              )
            )

        nef *= 2
        self.sp_model = nn.Sequential(
              nn.Linear(zsp_dim, nef), 
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(nef, nef), 
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(nef, nef), 
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(nef, 1)
            )

        if init_weights:
            self.init_weights()

    def forward(self, z):
        ##### z = z_fg + z_bg + z_sp
        zf, zb, zs = z[:,:self.zf_dim], \
                     z[:,self.zf_dim:self.zf_dim + self.zb_dim], \
                     z[:,-self.zsp_dim:]
        s_score = self.sp_model(zs)
        zf_logits = self.fg_model(zf)
        zb_logits = self.bg_model(zb)

        score = torch.logsumexp(zf_logits, dim=1, keepdim=True) + \
                torch.logsumexp(zb_logits, dim=1, keepdim=True) + \
                s_score
        return score, zf_logits, zb_logits