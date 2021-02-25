import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer as nr        
import numpy as np

###################################################################
####################### BASIC ARCH COMPONENTS #####################
###################################################################

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class GLU(nn.Module):
    """ GLU activation halves the channel number once applied """
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class UpBlock(nn.Module):
    """ upsample the feature map by a factor of 2x """
    def __init__(self, in_channels, out_channels, use_spc_norm=False):
        super(UpBlock, self).__init__()
        
        self.block = nn.Sequential(
              # nn.Upsample(scale_factor=2, mode='nearest'),
              nn.Upsample(scale_factor=2, mode='bilinear'),
              spectral_norm(
                nn.Conv2d(in_channels, 
                          out_channels, 
                          kernel_size=3, 
                          stride=1,
                          padding=1, 
                          bias=False),
                mode=use_spc_norm),
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
    def __init__(self, 
                 block_chns=[128, 1024, 512, 256, 128, 64],
                 use_spc_norm=False):
        super(BaseDecoder, self).__init__()
        blocks = []        
        for i in range(0, len(block_chns)-1):
            block = UpBlock(in_channels=block_chns[i], 
                            out_channels=block_chns[i+1],
                            use_spc_norm=use_spc_norm)
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

class T_Z(nn.Module):
    def __init__(self, z_dim):
        super(T_Z, self).__init__()
        self.enc = BaseEncoder(in_channels=3, 
                               out_channels=z_dim)
        self.layers = nn.Sequential(
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(z_dim + z_dim, 400),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(400, 400),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(400, 400),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(400, 1)
                        )

    def forward(self, x, z):
        bs = x.size(0)

        x = self.enc(x).view(bs, -1)
        return self.layers(torch.cat((x, z), dim=1))

class T_I(nn.Module):
    def __init__(self, z_dim):
        super(T_I, self).__init__()
        self.enc = BaseEncoder(in_channels=3, 
                               out_channels=z_dim)
        self.layers = nn.Sequential(
                          nn.Linear(z_dim + z_dim, 400),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(400, 400),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(400, 400),
                          nn.LeakyReLU(0.2, inplace=True),
                          nn.Linear(400, 1)
                        )

    def forward(self, fg, bg):
        bs = fg.size(0)

        h_f = self.enc(fg).view(bs, -1)
        h_b = self.enc(bg).view(bs, -1)
        return self.layers(torch.cat((h_f, h_b), dim=1))

class ImageHead(nn.Module):
    def __init__(self, in_channels, out_channels, use_spc_norm=False):
        super(ImageHead, self).__init__()        
        self.model = nn.Sequential(
            spectral_norm(
              nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=1, 
                      bias=True),
              mode=use_spc_norm),            
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class MaskHead(nn.Module):
    def __init__(self, in_channels, out_channels, use_spc_norm=False):
        super(MaskHead, self).__init__()        
        self.model = nn.Sequential(
            spectral_norm(
              nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=1,
                      padding=1, 
                      bias=True),
              mode=use_spc_norm),            
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
                       im_size=128,
                       use_spc_norm=False,
                       init_weights=True):
        super(FgNet, self).__init__(name=name)

        ############### generator arch ##############
        self.fc = nn.Sequential(
              spectral_norm(
                  nn.Linear(z_dim, block_chns[0] * 4 * 4, bias=True),
                  mode=use_spc_norm),
              nn.LeakyReLU(0.2, inplace=True)
            )

        # object feature base
        # z -> (B, 64, 128, 128)
        self.decode_base = BaseDecoder(block_chns=block_chns,
                                       use_spc_norm=use_spc_norm)

        self.im_head = ImageHead(in_channels=64, 
                                 out_channels=3,
                                 use_spc_norm=use_spc_norm)
        self.ma_head = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=64, 
                      out_channels=1, 
                      kernel_size=3, 
                      stride=1,
                      padding=1, 
                      bias=True),
                mode=use_spc_norm
              ))

        ############### encoder arch ##############
        # self.encode_base = BaseEncoder(in_channels=3, 
        #                                out_channels=z_dim,
        #                                use_spc_norm=False)
        if init_weights:
            self.init_weights()        

    def forward(self, z):
        bs = z.size(0)

        z_f = self.fc(z).view(bs, -1, 4, 4)
        obj_latent = self.decode_base(z_f)

        ### image 
        app = self.im_head(obj_latent)

        ### mask
        ma = self.ma_head(obj_latent)        

        ### logits for MI
        app_and_ma = app * ma.sigmoid() # .detach()
        
        return app, ma, app_and_ma, z_f

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
                       use_spc_norm=False,
                       init_weights=True):
        super(BgNet, self).__init__(name=name)

        ############### generator arch ##############
        self.fc = nn.Sequential(
              spectral_norm(
                  nn.Linear(z_dim, block_chns[0] * 4 * 4, bias=True),
                  mode=use_spc_norm),
              nn.LeakyReLU(0.2, inplace=True)
            )

        # object feature base
        # z -> (B, 64, 128, 128)
        self.decode_base = BaseDecoder(block_chns=block_chns,
                                       use_spc_norm=use_spc_norm)

        # final decoding layers
        self.im_head = ImageHead(
                          in_channels=64, 
                          out_channels=3,
                          use_spc_norm=use_spc_norm)
        self.ma_head = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=64, 
                      out_channels=1, 
                      kernel_size=3, 
                      stride=1,
                      padding=1, 
                      bias=True),
                mode=use_spc_norm
              ))

        ############### encoder arch ##############
        # self.encode_base = BaseEncoder(in_channels=3, 
        #                                out_channels=z_dim,
        #                                use_spc_norm=False)
        if init_weights:
            self.init_weights()

    def forward(self, z):
        bs = z.size(0)

        z_b = self.fc(z).view(bs, -1, 4, 4)            
        bg_latent = self.decode_base(z_b)

        ### image         
        bg = self.im_head(bg_latent)
        ### mask 
        ma = self.ma_head(bg_latent)

        ### logits for MI        
        bg_and_ma = bg * ma.sigmoid() # .detach()
        return bg, ma, bg_and_ma, z_b

class SpNet(BaseNetwork):
    def __init__(self, name='sp_net',
                       z_dim=128, 
                       zf_dim=128,
                       zb_dim=128,
                       im_size=128,     
                       block_chns=[128, 
                                   1024, 
                                   512, 
                                   256, 
                                   128, 
                                   64],
                       use_spc_norm=False,
                       init_weights=True):
        super(SpNet, self).__init__(name=name)
        self.im_size = im_size

        ############### generator arch ##############   
        self.fc = nn.Sequential(
              spectral_norm(
                  nn.Linear(z_dim, block_chns[0] * 4 * 4, bias=True),
                  mode=use_spc_norm),
              nn.LeakyReLU(0.2, inplace=True)
            )

        # feature base
        # z -> (B, 64, 128, 128)
        self.decode_base_deform = BaseDecoder(
                          block_chns=block_chns,
                          use_spc_norm=use_spc_norm)
        # deform grid est.
        self.decode_deform = ImageHead(64, 2, use_spc_norm)                

        if init_weights:
            self.init_weights()

    def forward(self, z, z_f, z_b):
        bs, d_z = z.size()        

        ############### bg deform estimation ##############
        z_sp = self.fc(z)
        deform_latent = z_sp.view(bs, -1, 4, 4)
        deform_latent = self.decode_base_deform(deform_latent)        
        deform_grid = self.decode_deform(deform_latent)
        # (B, 2, H, W) -> (B, H, W, 2)
        deform_grid = deform_grid.permute(0, 2, 3, 1)        

        return deform_grid, None, None # s_logits

###################################################################
########################## LATENT EBMS ############################
###################################################################

class CEBMNet(BaseNetwork):
    def __init__(self, name='ebm_net',
                       zf_dim=64,
                       zb_dim=32,
                       zsp_dim=128,
                       nef=200,
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
                nn.Linear(nef, nef),
                mode=use_spc_norm
              ),
              nn.LeakyReLU(0.2, inplace=True),
              spectral_norm(
                nn.Linear(nef, 1),
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
                nn.Linear(nef, nef),
                mode=use_spc_norm
              ),
              nn.LeakyReLU(0.2, inplace=True),
              spectral_norm(
                nn.Linear(nef, 1),
                mode=use_spc_norm
              )
            )

        nef *= 2
        self.sp_model = nn.Sequential(
              nn.Linear(zsp_dim + zb_dim, nef),
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
        zs_cat = torch.cat([zs, zb.detach()], dim=1)        
        zs_logits = self.sp_model(zs_cat)
        zf_logits = self.fg_model(zf)
        zb_logits = self.bg_model(zb)

        score = zf_logits + \
                zb_logits + \
                zs_logits
        return score, zf_logits, zb_logits, zs_logits

###################################################################
######################### DISCRIMINATOR ###########################
###################################################################

class DNet_bg(BaseNetwork):
    def __init__(self, 
                 in_channels=3, 
                 name='dnet_bg',
                 init_weights=True):
        super(DNet_bg, self).__init__(name=name)

        self.encode_base = nn.Sequential(
            # 128x128 -> 32x32
            DownBlock(in_channels, 64),
            DownBlock(64, 128),
            # shape-preserving
            SameBlock(128, 256, 0.2),
            SameBlock(256, 512, 0.2),            
        ) 

        self.real_fake_logits = nn.Sequential(            
            nn.Conv2d(
                    in_channels=512, 
                    out_channels=1, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False)
        )

        self.fg_bg_logits = nn.Sequential(            
            nn.Conv2d(
                    in_channels=512, 
                    out_channels=1, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        enc_latent = self.encode_base(x)        
        return self.real_fake_logits(enc_latent), \
               self.fg_bg_logits(enc_latent)

class DNet_all(BaseNetwork):
    def __init__(self, 
                 in_channels=3, 
                 name='dnet_all',
                 init_weights=True):
        super(DNet_all, self).__init__(name=name)

        self.model = nn.Sequential(
            # 128x128 -> 8x8
            DownBlock(in_channels, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
            nn.Conv2d(
                    in_channels=512, 
                    out_channels=1, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False)            
        ) 
        
        if init_weights:
            self.init_weights()

    def forward(self, x):
        return self.model(x)                