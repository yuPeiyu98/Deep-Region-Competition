import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer as nr        
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
    def __init__(self, in_channels, 
                       out_channels, 
                       use_spc_norm=False):
        super(UpBlock, self).__init__()
        
        self.block = nn.Sequential(
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

class UpDenseBlock(nn.Module):
    """ upsample the feature map by a factor of 2x """
    def __init__(self, in_channels, 
                       out_channels, 
                       use_spc_norm=False):
        super(UpDenseBlock, self).__init__()
        self.upscale = nn.Upsample(
              scale_factor=2, mode='bilinear')

        self.block = nn.Sequential(
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
        x_upscale = self.upscale(x)
        return torch.cat([self.block(x_upscale),
                          x_upscale], dim=1)

class SameBlock(nn.Module):
    """ shape-preserving feature transformation """
    def __init__(self, in_channels, out_channels, r=.01, use_spc_norm=False):
        super(SameBlock, self).__init__()
        
        self.block = nn.Sequential(
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

class BaseDenseDecoder(nn.Module):
    def __init__(self, 
                 block_chns=[128, 1024, 512, 256, 128, 64],
                 use_spc_norm=False):
        super(BaseDenseDecoder, self).__init__()
        blocks = []        
        for i in range(0, 3):
            block = UpBlock(in_channels=block_chns[i], 
                            out_channels=block_chns[i+1],
                            use_spc_norm=use_spc_norm)
            blocks.append(block)        

        block = UpDenseBlock(in_channels=block_chns[3], 
                            out_channels=block_chns[4],
                            use_spc_norm=use_spc_norm)
        blocks.append(block)
        for i in range(4, len(block_chns)-1):
            block = UpDenseBlock(
                            in_channels=block_chns[i]+block_chns[i-1], 
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

class MLP(nn.Module):
    def __init__(self, in_channels,                        
                       out_channels,
                       nef,
                       hi_layers=2,                       
                       use_spc_norm=False):
        super(MLP, self).__init__()
        blocks = [
                 spectral_norm(
                  nn.Linear(in_channels, nef),
                  mode=use_spc_norm
                 )]
        for __ in range(0, hi_layers):
            blocks += [
                       nn.LeakyReLU(0.2, inplace=True),
                       spectral_norm(
                        nn.Linear(nef, nef),
                        mode=use_spc_norm
                      )]
        blocks += [
                  nn.LeakyReLU(0.2, inplace=True),
                  spectral_norm(
                    nn.Linear(nef, out_channels),
                    mode=use_spc_norm
                  )]
        self.model = nn.Sequential(*blocks)  

    def forward(self, x):
        return self.model(x)

class DIMLP(nn.Module):
    def __init__(self, in_channels,                        
                       out_channels,
                       nef,                     
                       use_spc_norm=False):
        super(DIMLP, self).__init__()
        self.h1 = nn.Sequential(
                    spectral_norm(
                      nn.Linear(in_channels, nef),
                      mode=use_spc_norm),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(
                      nn.Linear(nef, nef),
                      mode=use_spc_norm)
                    )
        self.h2 = nn.Sequential(
                    spectral_norm(
                      nn.Linear(in_channels, nef),
                      mode=use_spc_norm),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(
                      nn.Linear(nef, nef),
                      mode=use_spc_norm)
                    )
        self.out = nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(
                        nn.Linear(nef * 2, nef),
                        mode=use_spc_norm),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(
                        nn.Linear(nef, out_channels),
                        mode=use_spc_norm)
                    )
                

    def forward(self, x1, x2):
        h1 = self.h1(x1)
        h2 = self.h2(x2)
        return self.out(torch.cat([h1, h2], dim=1))

class T_Z(nn.Module):
    def __init__(self, z_dim, 
                       Ks=[2,4,5,5],                       
                       use_spc_norm=False):
        super(T_Z, self).__init__()
        self.enc = BaseEncoder(in_channels=3, 
                               out_channels=z_dim,
                               use_spc_norm=use_spc_norm)
        self.mlps = nn.ModuleList([
                      MLP(z_dim, K,
                          nef=200,
                          hi_layers=1,
                          use_spc_norm=use_spc_norm)
                      for K in Ks])

    def forward(self, x):
        bs = x.size(0)

        z = self.enc(x).view(bs, -1)
        out = [mlp(z) for mlp in self.mlps]
        return out

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
        self.fc = nn.ModuleList([
                    nn.Sequential(
                      spectral_norm(
                          nn.Linear(128, 32 * 4 * 4, bias=True),
                          mode=use_spc_norm),
                      nn.LeakyReLU(0.2, inplace=True))
                    for __ in range(0, 4)
                  ])         

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
        # self.encode_base = T_Z(256, 256, 10, 50)
        if init_weights:
            self.init_weights()        

    def forward(self, z):
        bs = z.size(0)

        z_f = torch.cat([fc(z[:, int(128*i):int(128*(i+1))]).view(bs, -1, 4, 4) 
                         for i, fc in enumerate(self.fc)], 
                        dim=1)
        obj_latent = self.decode_base(z_f)

        ### image 
        app = self.im_head(obj_latent)

        ### mask
        ma = self.ma_head(obj_latent)        

        ### logits for MI
        app_and_ma = app * ma.sigmoid()
        # f_logits = self.encode_base(app_and_ma)
        
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
        self.fc = nn.ModuleList([
                    nn.Sequential(
                      spectral_norm(
                          nn.Linear(128, 32 * 4 * 4, bias=True),
                          mode=use_spc_norm),
                      nn.LeakyReLU(0.2, inplace=True))
                    for __ in range(0, 4)
                  ])   

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
        # self.encode_base = T_Z(256, 256, 10, 50)
        if init_weights:
            self.init_weights()

    def forward(self, z):
        bs = z.size(0)

        z_b = torch.cat([fc(z[:, int(128*i):int(128*(i+1))]).view(bs, -1, 4, 4) 
                         for i, fc in enumerate(self.fc)], 
                        dim=1)
        bg_latent = self.decode_base(z_b)

        ### image         
        bg = self.im_head(bg_latent)
        ### mask 
        ma = self.ma_head(bg_latent)

        ### logits for MI        
        # b_logits = self.encode_base(bg * ma.sigmoid())
        return bg, ma, bg * ma.sigmoid(), z_b

class SpNet(BaseNetwork):
    def __init__(self, name='sp_net',
                       z_dim=128,                        
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
        self.fc = nn.ModuleList([
                    nn.Sequential(
                      spectral_norm(
                          nn.Linear(128, 32 * 4 * 4, bias=True),
                          mode=use_spc_norm),
                      nn.LeakyReLU(0.2, inplace=True))
                    for __ in range(0, 4)
                  ])   

        # feature base
        # z -> (B, 64, 128, 128)
        # block_chns[0] += 128
        self.decode_base_deform = BaseDecoder(
                          block_chns=block_chns,
                          use_spc_norm=use_spc_norm)
        # deform grid est.
        self.decode_deform = ImageHead(64, 2, use_spc_norm)        
        
        if init_weights:
            self.init_weights()

    def forward(self, z):
        bs, d_z = z.size()        
 
        ############### bg deform estimation ##############
        z_sp = torch.cat([fc(z[:, int(128*i):int(128*(i+1))]).view(bs, -1, 4, 4) 
                         for i, fc in enumerate(self.fc)], 
                        dim=1)
        deform_latent = self.decode_base_deform(z_sp)        
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
                       nef=200,
                       Kf=[2,4,5,5],
                       Kb=[2,5,10,10],
                       use_spc_norm=False,
                       init_weights=True):
        super(CEBMNet, self).__init__(name=name)
        self.zf_dim = zf_dim
        self.zb_dim = zb_dim

        self.fg_model_0 = MLP(128, Kf[0],                               
                            nef=nef, 
                            hi_layers=1, 
                            use_spc_norm=use_spc_norm)
        self.fg_model_i = nn.ModuleList(                
                [DIMLP(128, K, 
                       nef=nef,
                       use_spc_norm=use_spc_norm)
                 for K in Kf[1:]])   

        self.bg_model_0 = MLP(128, Kb[0],                               
                            nef=nef, 
                            hi_layers=1, 
                            use_spc_norm=use_spc_norm)
        self.bg_model_i = nn.ModuleList(                
                [DIMLP(128, K, 
                       nef=nef,
                       use_spc_norm=use_spc_norm)
                 for K in Kb[1:]])  

        if init_weights:
            self.init_weights()

    def forward(self, z):
        ##### z = z_fg + z_bg + z_sp
        zf, zb = z[:,:self.zf_dim], \
                 z[:,self.zf_dim:self.zf_dim + self.zb_dim]

        zf_logits = [self.fg_model_0(zf[:, :128])] + \
                    [fg_model_i(zf[:, int(i*128):int((i+1)*128)].detach(),
                                zf[:, int((i+1)*128):int((i+2)*128)])
                        for i, fg_model_i in enumerate(self.fg_model_i)]
        zb_logits = [self.bg_model_0(zb[:, :128])] + \
                    [bg_model_i(zb[:, int(i*128):int((i+1)*128)].detach(),
                                zb[:, int((i+1)*128):int((i+2)*128)])
                        for i, bg_model_i in enumerate(self.bg_model_i)]

        score = 0
        for zf_logit, zb_logit in zip(zf_logits, zb_logits):
            score = score + zf_logit.logsumexp(dim=-1, keepdim=True) + \
                            zb_logit.logsumexp(dim=-1, keepdim=True)
        
        return score, zf_logits, \
                      zb_logits, None

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
