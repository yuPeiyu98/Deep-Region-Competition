import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .networks_gmm_amor import FgNet, BgNet, SpNet, CEBMNet
from .networks_gmm_amor import T_Z, FENet, BENet
from .loss import PerceptualLoss, StyleLoss, DiceLoss, GradLoss, SnakeLoss, SSIM
from .loss import tv_loss, bias_loss, dce, cce, ortho_reg

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

                torch.save({
                    'iteration': self.iteration,
                    module.name: module.state_dict()
                }, module_path)        

###################################################################
################## ALTERNATING BACKPROP W/ EBM ####################
###################################################################

class EABPModel(BaseModel):
    def __init__(self, config):
        super(EABPModel, self).__init__('EABPModel', config)
        self.use_spc_norm = False # True

        #####
        self.sigma = config.SIGMA
        self.sigma_z = 5e-1
        self.lambda_ = 1e-2

        self.delta_0 = config.DELTA_0
        self.infer_step_K0 = config.INFER_STEP_K0

        # Network configuration
        self.zf_dim = config.ZF_DIM
        self.zb_dim = config.ZB_DIM

        ### generator
        # fg
        self.fg_net = FgNet(z_dim=config.ZF_DIM,
                            im_size=config.IM_SIZE,
                            use_spc_norm=self.use_spc_norm)          
        self.fge_net = T_Z(256, Ks=[2,4,5,5], use_spc_norm=False)

        # bg
        self.bg_net = BgNet(z_dim=config.ZB_DIM,
                            use_spc_norm=self.use_spc_norm)
        self.bge_net = T_Z(256, Ks=[2,4,5,5], use_spc_norm=False)
        self.sp_net = SpNet(z_dim=config.ZS_DIM,
                            im_size=config.IM_SIZE,
                            use_spc_norm=self.use_spc_norm)        
        # ebm
        self.ebm_net = CEBMNet(zf_dim=config.ZF_DIM,
                               zb_dim=config.ZB_DIM,
                               Kb=[2,4,5,5],
                               nef=config.NEF,
                               use_spc_norm=False)        

        # inference
        self.fe_net = FENet(
            z_dim=256, use_spc_norm=self.use_spc_norm)
        self.be_net = BENet(
            z_dim=256, use_spc_norm=self.use_spc_norm)

        # Optims
        self.fg_optimizer = optim.Adam(
                params=[
                    {'params': self.fg_net.parameters()},
                    {'params': self.fge_net.parameters(),
                     'lr': float(config.LR)}],
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )        
        self.bg_optimizer = optim.Adam(
                params=[
                    {'params': self.bg_net.parameters()},
                    {'params': self.bge_net.parameters(),
                     'lr': float(config.LR)}],
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )
        self.sp_optimizer = optim.Adam(
                params=self.sp_net.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )  

        ### ebm      
        self.ebm_optimizer = optim.Adam(
                params=self.ebm_net.parameters(),
                lr=float(config.LR) * 1.,
                betas=(config.BETA1, config.BETA2)
            )

        ### inference
        self.fe_optimizer = optim.Adam(
                params=self.fe_net.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )        
        self.be_optimizer = optim.Adam(
                params=self.be_net.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )

    def save(self):
        # save models
        BaseModel.save(self)
        # save modified latent vars
        z_path = osp.join(self.BASE_PATH, '{}_z.pth'.format(
                                            self.model_name))        

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


    def _warp(self, aff_grid, deform_grid, fg, bg):
        warped_fg = fg         
        warped_bg = F.grid_sample(bg, deform_grid)
        return warped_fg, warped_bg

    def forward(self, z):
        zf, zb = z[:,:self.zf_dim], \
                 z[:,self.zf_dim:self.zf_dim + self.zb_dim]

        # generating background
        bg, bg_ma_logits, b_logits, bg_lat = self.bg_net(zb)
        # generating foreground & obj latent vec.
        fg, fg_ma_logits, f_logits, fg_lat = self.fg_net(zf)
        # generating spatial transformation (deforming grid & affine grid)
        d_grid, a_grid, s_logits = self.sp_net(zb)
        # image composition
        fg_wp, bg_wp = self._warp(a_grid, d_grid, fg, bg)
        fm_wp, bm_wp = self._warp(a_grid, d_grid, fg_ma_logits, bg_ma_logits)
        # fe_wp, be_wp = self._warp(a_grid, d_grid, fg_ed, bg_ed)

        pi = torch.cat([fm_wp, bm_wp], dim=1).softmax(dim=1)
        pi_f, pi_b = pi[:,:1,...], pi[:,1:,...]

        pi_f_ = torch.cat(
            [fm_wp, bm_wp.detach()], dim=1).softmax(dim=1)[:,:1,...]
        pi_b_ = torch.cat(
            [fm_wp.detach(), bm_wp], dim=1).softmax(dim=1)[:,1:,...]
        f_logits = pi_f_ * fg_wp
        b_logits = pi_b_ * bg_wp
        
        im_p = fg_wp * pi_f + bg_wp * pi_b
        return im_p, fg_wp, bg_wp, pi_f, fg, bg, pi_b, \
               f_logits, b_logits, s_logits, None, None

    def sample_langevin_prior(self, im_t):
        def _sample_p_0(bs, dim):
            return torch.randn(bs, dim)    
        bs = im_t.size(0)
        z = _sample_p_0(bs, self.zf_dim + \
                            self.zb_dim).to(im_t.device).requires_grad_(True)

        # langevin prior inference
        self._set_requires_grad(self.ebm_net, False)
        for __ in range(self.infer_step_K0):            
            en = self.ebm_net(z)[0]
            e_log_lkhd = en.sum() + self.sigma_z * z.square().sum()
            
            d_ebm_z = torch.autograd.grad(e_log_lkhd, z)[0]
            z.data = z.data - 0.5 * (self.delta_0 ** 2) * d_ebm_z \
                    + self.delta_0 * torch.randn_like(z).data
        
        self._set_requires_grad(self.ebm_net, True)
        return z.detach()

    def sample_posterior(self, im_t):        
        def reparametrize(mu, log_var):
            if self.training:
                std = torch.exp(log_var.mul(0.5))
                eps = torch.randn_like(std).to(mu.device)
                return eps.mul(std).add_(mu)
            else:
                return mu

        f_mu, f_var = self.fe_net(im_t)
        b_mu, b_var = self.be_net(im_t)

        z_f = [reparametrize(mu, var) for mu, var in zip(f_mu, f_var)]
        z_b = [reparametrize(mu, var) for mu, var in zip(b_mu, b_var)]
        z = torch.cat([torch.cat(z_f, dim=-1), torch.cat(z_b, dim=-1)],
                      dim=-1)
        return z, f_mu, f_var, b_mu, b_var

    def update_IG(self, im_t, 
                 z, f_mu, f_var, b_mu, b_var,
                 fg_wp, bg_wp, pi_f, pi_b, 
                 f_logits, b_logits,
                 zf_logits, zb_logits,
                 fe_wp, be_wp):
        bs = im_t.size(0)
        self._set_requires_grad(self.ebm_net, False)
        ##### jointly update generator and inference net
        self.fg_optimizer.zero_grad()
        self.bg_optimizer.zero_grad()
        self.sp_optimizer.zero_grad()

        self.fe_optimizer.zero_grad()
        self.be_optimizer.zero_grad()        

        ##### G loss
        # Mutual information                        
        fg_logits = self.fge_net(f_logits)
        bg_logits = self.bge_net(b_logits)

        fmi = torch.zeros(1).to(im_t.device)
        for zf_logit, fg_logit in zip(zf_logits, fg_logits):
            fmi = fmi + dce(zf_logit.detach(), fg_logit).mean()
                
        bmi = torch.zeros(1).to(im_t.device)
        for zb_logit, bg_logit in zip(zb_logits, bg_logits):
            bmi = bmi + dce(zb_logit.detach(), bg_logit).mean()

        # EM loss
        log_pf = - F.l1_loss(fg_wp, im_t, reduction='none')
        log_pb = - F.l1_loss(bg_wp, im_t, reduction='none')
        with torch.no_grad():
            ga_f = pi_f * log_pf.exp() / (pi_f * log_pf.exp() + pi_b * log_pb.exp() + 1e-8)

        e_z_log_p = ga_f.detach() * ((pi_f + 1e-8).log() + log_pf) \
                  + (1. - ga_f.detach()) * ((pi_b + 1e-8).log() + log_pb)
        recon_loss = -e_z_log_p.sum(dim=[2,3]).mean()

        # reg
        deform_pe = tv_loss(bg_wp).sum(dim=[2,3]).mean()
        G_reg = torch.zeros(1).to(z.device)
        # G_reg = ortho_reg(self.fg_net, im_t.device) + \
        #       ortho_reg(self.bg_net, im_t.device) + \
        #       ortho_reg(self.sp_net, im_t.device)

        G_loss = deform_pe * self.lambda_ + \
                 recon_loss + fmi * 1e-1 + bmi * 1e-1 + G_reg * 1e-1

        ##### I loss
        KLD = 0
        for mu, log_var in zip(f_mu + b_mu, f_var + b_var):
            KLD = KLD - 0.5 * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)        
        en = self.ebm_net(z)[0]
        I_reg = torch.zeros(1).to(z.device)
        # I_reg = ortho_reg(self.fe_net, im_t.device) + \
        #         ortho_reg(self.be_net, im_t.device)        

        I_loss = 1e-2 * KLD.mean() + en.mean() + I_reg * 1e-1

        IG_loss = I_loss + G_loss
        IG_loss.backward()

        self.fg_optimizer.step()
        self.bg_optimizer.step()
        self.sp_optimizer.step()
        
        self.fe_optimizer.step()
        self.be_optimizer.step()

        self._set_requires_grad(self.ebm_net, True)
        return deform_pe.item(), \
               fmi.item(), bmi.item(), G_reg.item(), \
               recon_loss.item(), KLD.mean().item()
    
    def update_E(self, en_pos, en_neg, 
                 zf_logits, 
                 zb_logits):
        self.ebm_optimizer.zero_grad()
        
        ebm_loss = en_pos.mean() - en_neg.mean()

        e_reg = torch.zeros(1).to(en_pos.device) 
        # ortho_reg(self.ebm_net, en_pos.device)

        loss = ebm_loss * 1.0 + e_reg * .1
        loss.backward()

        self.ebm_optimizer.step()

        return ebm_loss.item(), e_reg.item()

    def learn(self, zn, im_t):
        self.iteration += 1

        # posterior sampling
        zp, f_mu, f_var, b_mu, b_var = self.sample_posterior(im_t)

        im_p, fg_wp, bg_wp, pi_f, fg, bg, pi_b, \
               f_logits, b_logits, s_logits, fe_wp, be_wp = self(zp)

        en_pos, zpf_logits, zpb_logits, __ = self.ebm_net(zp.detach())
        en_neg, znf_logits, znb_logits, __ = self.ebm_net(zn)

        deform_pe, fmi, bmi, g_reg, \
        l1_loss, kld = self.update_IG(im_t, 
                                 zp, f_mu, f_var, b_mu, b_var,
                                 fg_wp, bg_wp, pi_f, pi_b, 
                                 f_logits, b_logits,
                                 zpf_logits, zpb_logits,
                                 fe_wp, be_wp)

        # update EBM in the latent space
        ebm_loss, e_reg = self.update_E(
                                en_pos, en_neg, 
                                znf_logits, 
                                znb_logits)

        with torch.no_grad():
            recon_loss = F.mse_loss(im_p, im_t).item()

        logs = [          
            ("recon_loss", recon_loss),            
            ("deform", deform_pe),
            ("fmi", fmi),
            ("bmi", bmi),
            ("g_reg", g_reg),
            ("ebm", ebm_loss),            
            ("kld", kld)     
        ]                      
        return logs
