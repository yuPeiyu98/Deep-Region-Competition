import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from .networks_gmm_att import FgNet, BgNet, SpNet, CEBMNet, DNet_bg, DNet_all
from .networks_gmm_sp import FgNet, BgNet, SpNet, CEBMNet, DNet_bg, DNet_all
from .loss import PerceptualLoss, StyleLoss, DiceLoss
from .loss import SnakeLoss, SSIM, tv_loss, bias_loss, dce, cce

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
        self.sigma = config.SIGMA
        self.sigma_z = .5
        self.lambda_ = 1e-2
        self.delta_0 = config.DELTA_0
        self.delta_1 = config.DELTA_1
        self.infer_step_K0 = config.INFER_STEP_K0
        self.infer_step_K1 = config.INFER_STEP_K1
        self.acm_step = config.ACM_STEP
        self.N = config.N_SAMPLE

        # Network configuration
        self.zf_dim = config.ZF_DIM
        self.zb_dim = config.ZB_DIM
        self.zs_dim = config.ZS_DIM

        self.fg_net = FgNet(z_dim=config.ZF_DIM,
                            im_size=config.IM_SIZE)
        self.bg_net = BgNet(z_dim=config.ZB_DIM)
        self.sp_net = SpNet(z_dim=config.ZS_DIM,
                            im_size=config.IM_SIZE)
        
        self.ebm_net = CEBMNet(zf_dim=config.ZF_DIM,
                               zb_dim=config.ZB_DIM,
                               zsp_dim=config.ZS_DIM,
                               nef=config.NEF)        

        # Optims
        self.fg_optimizer = optim.Adam(
                params=self.fg_net.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )
        self.bg_optimizer = optim.Adam(
                params=self.bg_net.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )
        self.sp_optimizer = optim.Adam(
                params=self.sp_net.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )        
        self.ebm_optimizer = optim.Adam(
                params=self.ebm_net.parameters(),
                lr=float(config.LR) * .2,
                betas=(config.BETA1, config.BETA2)
            )        

        # Latent vars initialization
        self.z_all_prior = torch.randn(self.N, self.zf_dim + \
                                               self.zb_dim + \
                                               self.zs_dim)
        self.z_all_poste = torch.randn(self.N, self.zf_dim + \
                                               self.zb_dim + \
                                               self.zs_dim)        

        # base grid
        self.b_grid = self._get_base_grid(config.IM_SIZE)

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
            
            self.z_all_prior = meta['z_all_prior'].cpu().requires_grad_(False)
            self.z_all_poste = meta['z_all_poste'].cpu().requires_grad_(False)

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

    def _get_base_grid(self, L):
        offset_y, offset_x = torch.meshgrid([torch.arange(L), torch.arange(L)])
        base_grid = torch.stack((offset_x, offset_y), dim=0).float()
        base_grid = 2. * base_grid / L - 1. + 1. / L
        return base_grid.permute(1, 2, 0)

    def _warp(self, aff_grid, deform_grid, fg, bg):
        # warped_fg = fg 
        warped_fg = F.grid_sample(fg, aff_grid)
        warped_bg = F.grid_sample(bg, deform_grid)
        return warped_fg, warped_bg

    def forward(self, z, im_t):
        zf, zb, zs = z[:,:self.zf_dim], \
                     z[:,self.zf_dim:self.zf_dim + self.zb_dim], \
                     z[:,-self.zs_dim:]

        # generating background
        bg, bg_ma_logits, b_logits, bg_lat = self.bg_net(zb)
        # generating foreground & obj latent vec.
        fg, fg_ma_logits, f_logits, fg_lat = self.fg_net(zf)
        # generating spatial transformation (deforming grid & affine grid)
        d_grid, a_grid, s_logits = self.sp_net(zs, fg_lat.detach(), bg_lat.detach())
        # image composition
        fg_wp, bg_wp = self._warp(a_grid, d_grid, fg, bg)
        fm_wp, bm_wp = self._warp(a_grid, d_grid, fg_ma_logits, bg_ma_logits)

        pi = torch.cat([fm_wp, bm_wp], dim=1).softmax(dim=1)
        pi_f, pi_b = pi[:,:1,...], pi[:,1:,...]

        with torch.no_grad():
            im_p = fg_wp * pi_f + bg_wp * pi_b
        return im_p, fg_wp, bg_wp, pi_f, fg, bg, pi_b, \
               f_logits, b_logits, s_logits, d_grid, a_grid

    def sample_langevin_prior(self, im_t, index):
        z = self.z_all_prior[index.cpu()].to(
                    im_t.device).requires_grad_(True)

        # langevin prior inference
        self._set_requires_grad(self.ebm_net, False)
        for __ in range(self.infer_step_K0):            
            en, __, __, __ = self.ebm_net(z)
            e_log_lkhd = en.sum() + self.sigma_z * z.square().sum()
            
            d_ebm_z = torch.autograd.grad(e_log_lkhd, z)[0]
            z.data = z.data - 0.5 * (self.delta_0 ** 2) * d_ebm_z \
                    + self.delta_0 * torch.randn_like(z).data
        
        self._set_requires_grad(self.ebm_net, True)
        self.z_all_prior[index.cpu()] = z.cpu()
        return z.detach()

    def sample_langevin_posterior(self, im_t, index=None):
        # z->z_f, z_b, z_o
        def _sample_p_0(bs, dim):
            return torch.randn(bs, dim)    
        bs = im_t.size(0)

        if self.training:
            z = self.z_all_poste[index.cpu()].to(im_t.device).requires_grad_(True)
            infer_step = self.infer_step_K1
        else:
            z = _sample_p_0(bs, self.zf_dim+\
                            self.zb_dim+\
                            self.zs_dim).to(im_t.device).requires_grad_(True)
            infer_step = 2500 
        
        self._set_requires_grad(self.fg_net, False)
        self._set_requires_grad(self.bg_net, False)
        self._set_requires_grad(self.sp_net, False)
        self._set_requires_grad(self.ebm_net, False)
        # langevin posterior inference
        for __ in range(infer_step):
            im_p, fg_wp, bg_wp, pi_f, fg, bg, pi_b, \
               f_logits, b_logits, s_logits, d_grid, a_grid = self(z, im_t)

            # energy est.
            en, __, __, __ = self.ebm_net(z)

            # reg
            deform_pe = tv_loss(bg_wp).sum(dim=[2,3]).sum()
            
            # posterior resp

            log_pf = - F.l1_loss(fg_wp, im_t, reduction='none') / (2. * self.sigma ** 2)
            log_pb = - F.l1_loss(bg_wp, im_t, reduction='none') / (2. * self.sigma ** 2)
            with torch.no_grad():
                ga_f = pi_f * log_pf.exp() / (pi_f * log_pf.exp() + pi_b * log_pb.exp() + 1e-8)

            e_z_log_p = ga_f.detach() * ((pi_f + 1e-8).log() + log_pf) \
                      + (1. - ga_f.detach()) * ((pi_b + 1e-8).log() + log_pb)

            j_log_lkhd = - e_z_log_p.sum() + \
                       en.sum() + self.sigma_z * z.square().sum() + \
                       deform_pe * self.lambda_

            d_j_z = torch.autograd.grad(j_log_lkhd, z)[0]
            z.data = z.data - 0.5 * (self.delta_1 ** 2) * d_j_z \
                   + self.delta_1 * torch.randn_like(z).data

        self._set_requires_grad(self.fg_net, True)
        self._set_requires_grad(self.bg_net, True)
        self._set_requires_grad(self.sp_net, True)
        self._set_requires_grad(self.ebm_net, True)
        # update z_all_posterior
        if self.training:            
            self.z_all_poste[index.cpu()] = z.cpu()                    
        return z.detach()

    def update_G(self, im_t, fg_wp, bg_wp, pi_f, pi_b, 
                 f_logits, b_logits, s_logits,
                 zf_logits, zb_logits, zs_logits,
                 d_grid, a_grid):
        ##### jointly update generator
        self.fg_optimizer.zero_grad()
        self.bg_optimizer.zero_grad()
        self.sp_optimizer.zero_grad()        

        # Mutual information        
        # fmi = torch.zeros(1).cuda() 
        fmi = dce(zf_logits, f_logits).mean()
        # bmi = torch.zeros(1).cuda() 
        bmi = dce(zb_logits, b_logits).mean()

        smi = torch.zeros(1).cuda() # cce(zs_logits, s_logits)  

        # reg
        deform_pe = tv_loss(bg_wp).mean() 
        # torch.zeros(1).to(im_t.device)                         

        log_pf = - F.l1_loss(fg_wp, im_t, reduction='none')
        log_pb = - F.l1_loss(bg_wp, im_t, reduction='none')
        with torch.no_grad():
            ga_f = pi_f * log_pf.exp() / (pi_f * log_pf.exp() + pi_b * log_pb.exp() + 1e-8)

        e_z_log_p = ga_f.detach() * ((pi_f + 1e-8).log() + log_pf) \
                  + (1. - ga_f.detach()) * ((pi_b + 1e-8).log() + log_pb)
        recon_loss = -e_z_log_p.mean() 
                       
        G_loss = deform_pe * self.lambda_ + \
                 recon_loss + fmi * 1e-1 + bmi * 1e-1 + smi * 1e-1

        G_loss.backward()
        self.fg_optimizer.step()
        self.bg_optimizer.step()
        self.sp_optimizer.step()

        return deform_pe.item(), \
               fmi.item(), bmi.item(), smi.item(), \
               recon_loss.item()
    
    def update_E(self, en_pos, en_neg, 
                 zf_logits, 
                 zb_logits):
        self.ebm_optimizer.zero_grad()
        
        ebm_loss = en_pos.mean() - en_neg.mean()
        # conditional entropy
        cond_ent_f = torch.zeros(1).cuda() # dce(zf_logits, zf_logits) 
        cond_ent_b = torch.zeros(1).cuda() # dce(zb_logits, zb_logits) 

        loss = ebm_loss * 1.0 + cond_ent_f * .01 + \
               cond_ent_b * .01
        loss.backward()

        self.ebm_optimizer.step()

        return ebm_loss.item(), cond_ent_f.item(), \
               cond_ent_b.item()

    def learn(self, zp, zn, im_t):
        self.iteration += 1

        im_p, fg_wp, bg_wp, pi_f, fg, bg, pi_b, \
               f_logits, b_logits, s_logits, d_grid, a_grid = self(zp, im_t)

        en_pos, zpf_logits, zpb_logits, __ = self.ebm_net(zp)
        zps_logits = zp[:,-self.zs_dim:]
        # en_pos, __, __, __ = self.ebm_net(zp)
        # zpf_logits, zpb_logits, zps_logits = zp[:,:self.zf_dim], \
        #                 zp[:,self.zf_dim:self.zf_dim + self.zb_dim], \
        #                 zp[:,-self.zs_dim:]
        en_neg, znf_logits, znb_logits, __ = self.ebm_net(zn)
        
        deform_pe, fmi, bmi, smi, \
        l1_loss = self.update_G(im_t, fg_wp, bg_wp, pi_f, pi_b,
                                f_logits, b_logits, s_logits,
                                zpf_logits.detach(), zpb_logits.detach(), zps_logits,
                                d_grid, a_grid)

        # update EBM in the latent space
        ebm_loss, cond_ent_f, cond_ent_b = self.update_E(
                                en_pos, en_neg, 
                                znf_logits, 
                                znb_logits)

        with torch.no_grad():
            recon_loss = F.mse_loss(im_p, im_t).item()

        logs = [
            # ("l_d_bg", dis_loss_bg),
            # ("l_d_bg_cls", errD_real_bg_cls),
            # ("l_d_all", dis_loss_a),
            ("l_g_recon", l1_loss),            
            ("recon_loss", recon_loss),            
            ("deform", deform_pe),
            ("fmi", fmi),
            ("bmi", bmi),
            ("smi", smi),
            ("ebm", ebm_loss),
            ("ent_f", cond_ent_f),
            ("ent_b", cond_ent_b)
        ]                      
        return logs
