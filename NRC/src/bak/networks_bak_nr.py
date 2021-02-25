import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer as nr        

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

class UpBlock(nn.Module):
    """ upsample the feature map by a factor of 2x """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.block = nn.Sequential(
              nn.Upsample(scale_factor=2, mode='nearest'),
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
    def __init__(self, in_channels, out_channels, r=1e-2):
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
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        
        self.block = nn.Sequential(
              nn.Conv2d(in_channels, 
                        out_channels, 
                        kernel_size=4, 
                        stride=2,
                        padding=1, 
                        bias=False),
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

    def init_weights(self, init_type='xavier', gain=.02):
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
                       block_chns=[64, 128, 256, 512, 1024]):
        super(BaseEncoder, self).__init__()
        blocks = [] 
        blocks.append(DownBlock(in_channels=in_channels,
                                out_channels=block_chns[0]))
        for i in range(0, len(block_chns)-1):
            block = DownBlock(in_channels=block_chns[i], 
                            out_channels=block_chns[i+1])
            blocks.append(block)        
        blocks.append(nn.Conv2d(in_channels=block_chns[-1],
                                out_channels=out_channels,
                                kernel_size=6,
                                stride=4,
                                padding=1,
                                bias=True))
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
                       npts=32,
                       im_size=128,
                       init_weights=True):
        super(FgNet, self).__init__(name=name)

        ############### generator arch ##############
        self.fc = nn.Sequential(
              nn.Linear(z_dim, block_chns[0] * 4 * 4, bias=True),
              nn.LeakyReLU(inplace=True)
            )

        # object feature base
        # z -> (B, 64, 128, 128)
        self.decode_base = BaseDecoder(block_chns=block_chns)

        # final decoding layers
        blocks = [ResBlock(block_chns[-1])]
        blocks_im = blocks + [ImageHead(in_channels=64, out_channels=3)]
        blocks_ma = blocks + [ImageHead(in_channels=64, out_channels=2)]
        self.im_head = nn.Sequential(*blocks_im)
        self.ma_head = nn.Sequential(*blocks_ma)
        # self.im_head = ImageHead(in_channels=64, out_channels=3)
        # self.ma_head = ImageHead(in_channels=64, out_channels=2)

        ############### encoder arch ##############
        self.encode_base = BaseEncoder(in_channels=3, 
                                       out_channels=kf)
        if init_weights:
            self.init_weights()

        ################ renderer #################
        self.renderer = nr.Renderer(
            camera_mode='look_at', 
            image_size=im_size, 
            light_intensity_ambient=1,
            light_intensity_directional=1, 
            perspective=False)

        self.prior_p, self.prior_f = self.get_poly(
                dim=im_size, n=npts, R=int(im_size*(3/8)), 
                xx=im_size//2, yy=im_size//2)
        self.max_iter = 3
        

    def get_poly(self, dim=32, n=16, R=12, xx=16, yy=16):
        import math
        import numpy as np
        from scipy.spatial import Delaunay

        half_dim = dim / 2
        P = [np.array([xx + math.floor(math.cos(2 * math.pi / n * x) * R),
                       yy + math.floor(math.sin(2 * math.pi / n * x) * R)]) \
                  for x in range(0, n)]
        train_data = torch.zeros(1, 1, n, 2)
        for i in range(n):
            train_data[0, 0, i, 0] = torch.tensor(
              (P[i][0] - half_dim) / half_dim).clone()
            train_data[0, 0, i, 1] = torch.tensor(
              (P[i][1] - half_dim) / half_dim).clone()
        vertices = torch.ones((n, 3))
        tmp = train_data.squeeze(dim=0).squeeze(dim=0)
        vertices[:, 0] = tmp[:, 0]
        vertices[:, 1] = tmp[:, 1] * -1
        tri = Delaunay(vertices[:, 0:2].numpy())
        faces = torch.tensor(tri.simplices.copy())
        return train_data, faces[None, None, ...]

    def render(self, dx, dy):
        bs = dx.size(0)

        P, Faces = self.prior_p, self.prior_f
        P = P.to(dx.device).repeat(bs, 1, 1, 1)
        Faces = Faces.to(dy.device).repeat(bs, 1, 1, 1)

        for i in range(0, self.max_iter):
            Pxx = F.grid_sample(dx, P).transpose(3, 2)
            Pyy = F.grid_sample(dy, P).transpose(3, 2)
            Pedge = torch.cat((Pxx, Pyy), -1)
            P = Pedge + P

        ##### render
        z = torch.ones((P.shape[0], 1, P.shape[2], 1)).to(dx.device)
        PP = torch.cat((P, z), 3)
        PP = torch.squeeze(PP, dim=1)
        PP[:, :, 1] = PP[:, :, 1]*-1
        faces = torch.squeeze(Faces, dim=1)
        self.renderer.eye = nr.get_points_from_angles(
            1, # camera_distance 
            0, # elevation, 
            0) # azimuth
        mask = self.renderer(PP, faces, mode='silhouettes').unsqueeze(dim=1)
        PP[:, :, 1] = PP[:, :, 1]*-1
        P_f = PP[:, :, 0:2].unsqueeze(dim=1)
        
        return mask, P_f, dx, dy

    def forward(self, z):
        bs = z.size(0)

        z_f = self.fc(z).view(bs, -1, 4, 4)
        obj_latent = self.decode_base(z_f)

        ### image 
        app = self.im_head(obj_latent)

        ### displacement map
        shift_map = self.ma_head(obj_latent)
        dx, dy = shift_map[:, :1, ...], \
                 shift_map[:, 1:, ...]

        mask, P, dx, dy = self.render(dx, dy)

        ### logits for MI
        app_and_ma = mask * app
        f_logits = self.encode_base(app_and_ma)
        f_logits = f_logits.view(bs, -1)
        
        return app, mask, P, f_logits, z_f

class BgNet(BaseNetwork):
    def __init__(self, name='bg_net',
                       z_dim=32,
                       block_chns=[128, 
                                   1024, 
                                   512, 
                                   256, 
                                   128, 
                                   64],
                       kb=20,
                       init_weights=True):
        super(BgNet, self).__init__(name=name)

        ############### generator arch ##############
        self.fc = nn.Sequential(
              nn.Linear(z_dim, block_chns[0] * 4 * 4, bias=True),
              nn.LeakyReLU(inplace=True)
            )

        # object feature base
        # z -> (B, 64, 128, 128)
        self.decode_base = BaseDecoder(block_chns=block_chns)        

        # final decoding layers
        self.im_head = ImageHead(in_channels=64, out_channels=3)

        ############### encoder arch ##############
        self.encode_base = BaseEncoder(in_channels=3, 
                                       out_channels=kb)
        if init_weights:
            self.init_weights()

    def forward(self, z):
        bs = z.size(0)

        z_b = self.fc(z).view(bs, -1, 4, 4)            
        bg_latent = self.decode_base(z_b)

        ### image         
        bg = self.im_head(bg_latent)

        ### logits for MI        
        b_logits = self.encode_base(bg)
        b_logits = b_logits.view(bs, -1)
        return bg, b_logits

class SpNet(BaseNetwork):
    def __init__(self, name='sp_net',
                       z_dim=128, 
                       zobj_dim=128,
                       im_size=128,     
                       num_res_blocks=3,                  
                       block_chns=[128, 
                                   1024, 
                                   512, 
                                   256, 
                                   128, 
                                   64],
                       init_weights=True):
        super(SpNet, self).__init__(name=name)
        self.im_size = im_size

        ############### generator arch ##############   
        self.fc = nn.Sequential(
              nn.Linear(z_dim, block_chns[0] * 4 * 4, bias=True),
              nn.LeakyReLU(inplace=True)
            )

        ### affine grid est.
        self.decode_base_aff = nn.Sequential(
                        nn.Linear(block_chns[0] * 4 * 4, 256),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(256, 128),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(128, 64),
                        nn.LeakyReLU(inplace=True)
                    )
        self.decode_t = nn.Sequential(
                        nn.Linear(64, 2),
                        nn.Tanh()
                    ) 
        self.decode_a = nn.Sequential(
                        nn.Linear(64, 4),
                        nn.Sigmoid()
                    )

        ### deform grid est.
        # z -> (B, 64, 128, 128)
        block_chns[0] += zobj_dim
        self.decode_base_deform = BaseDecoder(block_chns=block_chns)
        
        # deform grid head
        self.decode_deform = ImageHead(64, 2)

        ############### encoder arch ##############
        self.encode_base = BaseEncoder(in_channels=4, 
                                       out_channels=z_dim * 2)

        if init_weights:
            self.init_weights()

    def forward(self, z, obj_latent):
        bs = z.size(0)

        z_sp = self.fc(z)

        ############### affine estimation ##############        
        # 1. estimate affine matrix
        affine_latent = self.decode_base_aff(z_sp)
        aff = self.decode_a(affine_latent).view(bs, 2, 2)
        trs = self.decode_t(affine_latent).view(bs, 2, 1) * .5
        theta = torch.cat([aff, trs], dim=2)

        bias = torch.eye(2, 3).view(1, 2, 3).repeat(bs, 1, 1) * .5
        bias = bias.to(z.device)
        theta = theta + bias
        # 2. construct sampling grid
        out_shape = (bs, 3, self.im_size, self.im_size)
        aff_grid = F.affine_grid(theta, torch.Size(out_shape))
        
        ############### deform estimation ##############        
        deform_latent = torch.cat([
            obj_latent, z_sp.view(bs, -1, 4, 4)], dim=1)
        deform_latent = self.decode_base_deform(deform_latent)
        deform_grid = self.decode_deform(deform_latent)
        # (B, 2, H, W) -> (B, H, W, 2)
        deform_grid = deform_grid.permute(0, 2, 3, 1)
        
        ### logits for MI
        cat_grid = torch.cat([deform_grid, aff_grid], dim=-1)
        # (B, H, W, 4) -> (B, 4, H, W)
        s_logits = self.encode_base(cat_grid.permute(0, 3, 1, 2))
        s_logits = s_logits.view(bs, -1)
        return deform_grid, aff_grid, s_logits

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
                       Kb=20,
                       init_weights=True):
        super(CEBMNet, self).__init__(name=name)
        self.zf_dim = zf_dim
        self.zb_dim = zb_dim
        self.zsp_dim = zsp_dim
        
        self.bg_model = nn.Sequential(
              nn.Linear(zb_dim, nef),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(nef, nef), 
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(nef, Kb)
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
        self.fg_model = nn.Sequential(
              nn.Linear(zf_dim, nef), 
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(nef, nef), 
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(nef, nef), 
              nn.LeakyReLU(0.2, inplace=True),
              nn.Linear(nef, Kf)
            )

        if init_weights:
            self.init_weights(init_type='xavier', gain=.02)

    def forward(self, z):
        ##### z = z_fg + z_bg + z_sp
        zf, zb, zs = z[:,:self.zf_dim], \
                     z[:,self.zf_dim:self.zf_dim + self.zb_dim], \
                     z[:,-self.zsp_dim:]
        zs_logits = self.sp_model(zs)
        zf_logits = self.fg_model(zf)
        zb_logits = self.bg_model(zb)

        score = torch.logsumexp(zf_logits, dim=1, keepdim=True) + \
                torch.logsumexp(zb_logits, dim=1, keepdim=True) + \
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
            SameBlock(128, 256, .2),
            SameBlock(256, 512, .2),            
        ) 

        self.real_fake_logits = nn.Sequential(            
            nn.Conv2d(
                    in_channels=512, 
                    out_channels=1, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=True)
        )

        self.fg_bg_logits = nn.Sequential(            
            nn.Conv2d(
                    in_channels=512, 
                    out_channels=1, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=True)
        )

        if init_weights:
            self.init_weights(init_type='xavier', gain=.02)

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
                    bias=True)            
        ) 
        
        if init_weights:
            self.init_weights(init_type='xavier', gain=.02)

    def forward(self, x):
        return self.model(x)                
