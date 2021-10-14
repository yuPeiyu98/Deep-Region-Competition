import numpy as np
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .networks import FgNet, BgNet, SpNet, CEBMNet
from .loss import dce, tv_loss , ortho_reg

###################################################################
####################### BASE MODEL & UTILS ########################
###################################################################

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.model_name = name
        self.config = config        

        self.BASE_PATH = config.PATH
        self.iteration = 0

    def load(self):
        """ Retrieve saved modules """
        for __, module in enumerate(self.modules()):
            if hasattr(module, 'name'):
                module_path = osp.join(self.BASE_PATH, 
                                       '{}_{}.pth'.format(
                                                    self.model_name,
                                                    module.name))
                if os.path.exists(module_path):
                    print('Loading {} {}...'.format(self.model_name, 
                                                    module.name))

                    if torch.cuda.is_available():
                        meta = torch.load(module_path)
                    else:
                        meta = torch.load(
                                module_path, 
                                map_location=lambda storage, 
                                                loc: storage)

                    module.load_state_dict(meta[module.name])
                    self.iteration = meta['iteration']        

    def save(self):
        print('\nsaving {}...\n'.format(self.model_name))
        for __, module in enumerate(self.modules()):
            if hasattr(module, 'name'):
                module_path = osp.join(
                                self.BASE_PATH, 
                                '{}_{}.pth'.format(
                                              self.model_name,
                                              module.name))                

                # copy
                module_path_iter = osp.join(
                                self.BASE_PATH, 
                                '{}_{}_{}.pth'.format(
                                              self.model_name,
                                              module.name,
                                              self.iteration))                

                torch.save({
                    'iteration': self.iteration,
                    module.name: module.state_dict()
                }, module_path)        

                torch.save({
                    'iteration': self.iteration,
                    module.name: module.state_dict()
                }, module_path_iter)        

###################################################################
################## ALTERNATING BACKPROP W/ EBM ####################
###################################################################

class EABPModel(BaseModel):
    def __init__(self, config):
        super(EABPModel, self).__init__('EABPModel', config)
        self.sigma = config.SIGMA
        self.delta_0 = config.DELTA_0
        self.delta_1 = config.DELTA_1
        self.infer_step_K0 = config.INFER_STEP_K0
        self.infer_step_K1 = config.INFER_STEP_K1
        self.infer_test = config.INFER_TEST

        self.N = config.N_SAMPLE

        # Network configuration
        self.zf_dim = config.ZF_DIM
        self.zb_dim = config.ZB_DIM
        self.zs_dim = config.ZS_DIM

        self.fg_net = FgNet(z_dim=config.ZF_DIM)
        self.bg_net = BgNet(z_dim=config.ZB_DIM)
        self.sp_net = SpNet(z_dim=config.ZS_DIM)
        
        self.ebm_net = CEBMNet(zf_dim=config.ZF_DIM,
                               zb_dim=config.ZB_DIM,
                               zsp_dim=config.ZS_DIM,
                               nef=config.NEF)

        # Optims
        self.gen_optimizer = optim.Adam(
                params=[
                    {'params': self.fg_net.parameters()},
                    {'params': self.bg_net.parameters()},
                    {'params': self.sp_net.parameters()}
                ],
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )        
        self.ebm_optimizer = optim.Adam(
                params=self.ebm_net.parameters(),
                lr=float(config.LR) * .2,
                betas=(config.BETA1, config.BETA2)
            )

        # Latent vars initialization
        self.register_buffer(
            'z_all_prior',
            torch.randn(self.N, self.zf_dim + \
                                self.zb_dim + \
                                self.zs_dim)
        )
        self.register_buffer(
            'z_all_poste',
            torch.randn(self.N, self.zf_dim + \
                                self.zb_dim + \
                                self.zs_dim)   
        )

    def save(self):
        # save models
        BaseModel.save(self)
        # save modified latent vars
        z_path = osp.join(self.BASE_PATH, '{}_z.pth'.format(
                                            self.model_name))
        torch.save({
            'z_all_prior': self.z_all_prior,
            'z_all_poste': self.z_all_poste
        }, z_path) 

    def load(self):
        # load models       
        BaseModel.load(self)
        # load modified latent vars
        z_path = osp.join(self.BASE_PATH, '{}_z.pth'.format(
                                            self.model_name))
        if os.path.exists(z_path):
            if torch.cuda.is_available():
                meta = torch.load(z_path)
            else:
                meta = torch.load(z_path, 
                        map_location=lambda storage, 
                        loc: storage)
            
            self.z_all_prior = meta['z_all_prior'].requires_grad_(False)
            self.z_all_poste = meta['z_all_poste'].requires_grad_(False)

    def _set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to 
           avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks 
                                     require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad        

    def forward(self, z, im_t=None):
        zf, zb, zs = z[:,:self.zf_dim], \
                     z[:,self.zf_dim:self.zf_dim + self.zb_dim], \
                     z[:,-self.zs_dim:]

        # generating background
        bg, bg_ma_logits, b_logits, bg_lat = self.bg_net(zb)
        # generating foreground & obj latent vec.
        fg, fg_ma_logits, f_logits, fg_lat = self.fg_net(zf)
        # generating spatial transformation (deforming grid & affine grid)
        d_grid = self.sp_net(zs, bg_lat.detach())
        # image composition
        bg_wp = F.grid_sample(bg, d_grid)
        bm_wp = F.grid_sample(bg_ma_logits, d_grid)

        pi = torch.cat([fg_ma_logits, bm_wp], dim=1).softmax(dim=1)
        pi_f, pi_b = pi[:,:1,...], pi[:,1:,...]

        with torch.no_grad():
            im_p = fg * pi_f + bg_wp * pi_b
        return im_p, fg, bg_wp, pi_f, pi_b, bg, d_grid, \
               f_logits, b_logits

    def sample_langevin_prior(self, index=None):
        def _sample_p_0(bs, dim, device):            
            return torch.randn(bs, dim, device=device)    

        if self.training:
            z = self.z_all_prior[index]
            infer_step = self.infer_step_K0
        else:
            z = _sample_p_0(
                    index.size(0), self.zf_dim+\
                                   self.zb_dim+\
                                   self.zs_dim, index.device)
            infer_step = self.infer_test     

        # langevin prior inference
        self._set_requires_grad(self.ebm_net, requires_grad=False)
        for __ in range(self.infer_step_K0):
            z = z.requires_grad_(True)
            en, __, __ = self.ebm_net(z)
            e_log_lkhd = en.sum() + .5 * z.square().sum()
            
            d_ebm_z = torch.autograd.grad(e_log_lkhd, z)[0]
            z = z - 0.5 * (self.delta_0 ** 2) * d_ebm_z \
                    + self.delta_0 * torch.randn_like(z)
            z = z.detach()
        
        self._set_requires_grad(self.ebm_net, requires_grad=True)
        if self.training:
            self.z_all_prior[index] = z
        return z

    def sample_langevin_posterior(self, im_t, index=None):
        # z->z_f, z_b, z_o
        def _sample_p_0(bs, dim, device):
            return torch.randn(bs, dim, device=device)    
        bs = im_t.size(0)

        if self.training:
            z = self.z_all_poste[index]
            infer_step = self.infer_step_K1
        else:
            z = _sample_p_0(
                    bs, self.zf_dim+\
                        self.zb_dim+\
                        self.zs_dim, im_t.device)
            infer_step = self.infer_test
        
        # langevin posterior inference
        self._set_requires_grad(self.fg_net, requires_grad=False)
        self._set_requires_grad(self.bg_net, requires_grad=False)
        self._set_requires_grad(self.sp_net, requires_grad=False)
        self._set_requires_grad(self.ebm_net, requires_grad=False)
        for __ in range(infer_step):
            z = z.requires_grad_(True)
            im_p, fg, bg_wp, pi_f, pi_b, bg, d_grid, \
                f_logits, b_logits = self(z, im_t)

            # log-lkhd for lebms
            en, __, __ = self.ebm_net(z)
            

            # log-lkhd for generators
            log_pf = - F.l1_loss(fg, im_t, reduction='none') \
                     / (2. * self.sigma ** 2)
            log_pb = - F.l1_loss(bg_wp, im_t, reduction='none') \
                     / (2. * self.sigma ** 2)

            # posterior responsibilities
            with torch.no_grad():
                ga_f = pi_f * log_pf.exp() / \
                      (pi_f * log_pf.exp() + pi_b * log_pb.exp() + 1e-8)

            e_z_log_p = ga_f.detach() * \
                        ((pi_f + 1e-8).log() + log_pf) \
                      + (1. - ga_f.detach()) * \
                        ((pi_b + 1e-8).log() + log_pb)

            # regularization
            tv_norm = tv_loss(bg_wp).sum(dim=[2,3]).sum()                        

            j_log_lkhd = - e_z_log_p.sum() + tv_norm * .01 + \
                           en.sum() + .5 * z.square().sum()

            d_j_z = torch.autograd.grad(j_log_lkhd, z)[0]
            z = z - 0.5 * (self.delta_1 ** 2) * d_j_z \
                  + self.delta_1 * torch.randn_like(z)
            z = z.detach()

        # update z_all_posterior
        if self.training:            
            self.z_all_poste[index] = z
        self._set_requires_grad(self.fg_net, requires_grad=True)
        self._set_requires_grad(self.bg_net, requires_grad=True)
        self._set_requires_grad(self.sp_net, requires_grad=True)
        self._set_requires_grad(self.ebm_net, requires_grad=True)
        return z    

    def update_G(
        self, im_t, fg_wp, bg_wp, pi_f, pi_b, 
              f_logits, b_logits,
              zf_logits, zb_logits
        ):
        self.gen_optimizer.zero_grad()        

        ### regularization
        # pseudo-label learning 
        hpq_f = dce(zf_logits, f_logits).mean() 
        hpq_b = dce(zb_logits, b_logits).mean()

        # orthogonal regularizations
        ortho_regl = ortho_reg(self.fg_net, im_t.device) + \
                     ortho_reg(self.bg_net, im_t.device)

        # tv-norm
        tv_norm = tv_loss(bg_wp).mean() 

        ### log-lkhd for generators
        log_pf = - F.l1_loss(fg_wp, im_t, reduction='none')
        log_pb = - F.l1_loss(bg_wp, im_t, reduction='none')
        with torch.no_grad():
            ga_f = pi_f * log_pf.exp() / (pi_f * log_pf.exp() + pi_b * log_pb.exp() + 1e-8)

        e_z_log_p = (ga_f.detach() * ((pi_f + 1e-8).log() + log_pf) \
                  + (1. - ga_f.detach()) * ((pi_b + 1e-8).log() + log_pb)).mean()
                       
        G_loss = -e_z_log_p.mean() + tv_norm * .01 \
               + hpq_f * 1e-1 + hpq_b * 1e-1 + ortho_regl * 1e-1

        G_loss.backward()
        self.gen_optimizer.step()        

        return hpq_f.item(), hpq_b.item(), ortho_regl.item(), tv_norm.item(), \
               e_z_log_p.item()

    def update_E(self, en_pos, en_neg):
        self.ebm_optimizer.zero_grad()
        
        ebm_loss = en_pos.mean() - en_neg.mean()
        ebm_loss.backward()

        self.ebm_optimizer.step()

        return ebm_loss.item()

    def learn(self, im_t, index):
        self.iteration += 1

        ### 0. sample latent vectors
        zp = self.sample_langevin_posterior(im_t, index)
        zn = self.sample_langevin_prior(index)

        ### 1. update network parameters
        im_p, fg, bg_wp, pi_f, pi_b, bg, d_grid, \
                f_logits, b_logits = self(zp, im_t)        

        en_pos, zpf_logits, zpb_logits = self.ebm_net(zp)        
        en_neg, znf_logits, znb_logits = self.ebm_net(zn)

        # D requires no gradients when optimizing G        
        hpq_f, hpq_b, ortho_regl, tv_norm, e_z_log_p = self.update_G(
                                        im_t,
                                        fg, bg_wp,
                                        pi_f, pi_b,
                                        f_logits, b_logits,
                                        zpf_logits.detach(),
                                        zpb_logits.detach())

        # update EBM in the latent space
        ebm_loss = self.update_E(en_pos, en_neg)

        with torch.no_grad():
            recon_loss = F.mse_loss(im_p, im_t).item()

        logs = [
            ("recon_err", recon_loss),
            ("tv_norm", tv_norm),
            ("hpq_f", hpq_f),
            ("hpq_b", hpq_b),
            ("ortho_reg", ortho_regl),
            ("ebm_lss", ebm_loss)
        ]                      
        return logs
