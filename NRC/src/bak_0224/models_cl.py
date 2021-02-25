import math
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from .networks_gmm_att import FgNet, BgNet, SpNet, CEBMNet, DNet_bg, DNet_all
from .networks_cl import FgNet, BgNet, SpNet, CEBMNet, DNet_bg, DNet_all
from .networks_cl import T_Z, T_I, BaseEncoder
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
        self.use_spcn = True

        self.sigma = config.SIGMA
        self.sigma_zf = .25
        self.sigma_zb = .25
        self.sigma_zs = .5
        self.beta = .999
        self.Kf = 200
        self.Kb = 200
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
                            im_size=config.IM_SIZE,
                            use_spc_norm=self.use_spcn)
        self.fge_net = BaseEncoder(
                            in_channels=3, 
                            out_channels=self.Kf,
                            use_spc_norm=self.use_spcn)
        self.bg_net = BgNet(z_dim=config.ZB_DIM,
                            use_spc_norm=self.use_spcn)
        self.bge_net = BaseEncoder(
                            in_channels=3, 
                            out_channels=self.Kb,
                            use_spc_norm=self.use_spcn)

        self.sp_net = SpNet(z_dim=config.ZS_DIM,
                            im_size=config.IM_SIZE)

        self.ebm_net = CEBMNet(zf_dim=config.ZF_DIM,
                               zb_dim=config.ZB_DIM,
                               zsp_dim=config.ZS_DIM,
                               nef=config.NEF,
                               use_spc_norm=self.use_spcn)        

        # Optims
        self.fg_optimizer = optim.Adam(
                params=[
                    {'params': self.fg_net.parameters()},
                    {'params': self.fge_net.parameters()}
                ],
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)
            )
        self.bg_optimizer = optim.Adam(
                params=[
                    {'params': self.bg_net.parameters()},
                    {'params': self.bge_net.parameters()}
                ],
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
        # centroid init
        self.centroid_f = torch.randn(self.Kf, self.zf_dim)
        self.centroid_b = torch.randn(self.Kb, self.zb_dim)

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
            'z_all_poste': self.z_all_poste,
            'centroid_f': self.centroid_f,
            'centroid_b': self.centroid_b
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
            self.centroid_f = meta['centroid_f'].cpu().requires_grad_(False)
            self.centroid_b = meta['centroid_b'].cpu().requires_grad_(False)

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
        warped_fg = fg
        # warped_fg = F.grid_sample(fg, aff_grid)
        # warped_bg = bg
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
               f_logits, b_logits, s_logits, fg_ma_logits, bg_ma_logits

    def sample_langevin_prior(self, im_t, index):
        z = self.z_all_prior[index.cpu()].to(
                    im_t.device).requires_grad_(True)

        self._set_requires_grad(self.ebm_net, False)
        # langevin prior inference
        for __ in range(self.infer_step_K0):
            en, __, __, __ = self.ebm_net(z)
            zf, zb, zs = z[:,:self.zf_dim], \
                         z[:,self.zf_dim:self.zf_dim + self.zb_dim], \
                         z[:,-self.zs_dim:]
            e_log_lkhd = en.sum() + self.sigma_zf * zf.square().sum() + self.sigma_zb * zb.square().sum() \
                       + self.sigma_zs * zs.square().sum()

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

        # langevin posterior inference
        self._set_requires_grad(self.fg_net, False)
        self._set_requires_grad(self.bg_net, False)
        self._set_requires_grad(self.sp_net, False)
        self._set_requires_grad(self.ebm_net, False)
        for __ in range(infer_step):
            im_p, fg_wp, bg_wp, pi_f, fg, bg, pi_b, \
               f_logits, b_logits, s_logits, fg_ma_logits, bg_ma_logits = self(z, im_t)            

            # posterior resp (fg / bg)
            log_pf = - F.l1_loss(fg_wp, im_t, reduction='none') / (2. * self.sigma ** 2)
            log_pb = - F.l1_loss(bg_wp, im_t, reduction='none') / (2. * self.sigma ** 2)
            with torch.no_grad():
                ga_f = pi_f * log_pf.exp() / (pi_f * log_pf.exp() + pi_b * log_pb.exp() + 1e-8)

            e_z_log_p = ga_f.detach() * ((pi_f + 1e-8).log() + log_pf) \
                      + (1. - ga_f.detach()) * ((pi_b + 1e-8).log() + log_pb)

            # posterior resp (K class)
            zf, zb, zs = z[:,:self.zf_dim], \
                         z[:,self.zf_dim:self.zf_dim + self.zb_dim], \
                         z[:,-self.zs_dim:]
            cf = self.centroid_f.to(im_t.device).requires_grad_(False)
            cb = self.centroid_b.to(im_t.device).requires_grad_(False)
            # (B, K, D)
            zf_q_cf = zf.view(bs, 1, self.zf_dim) - cf.view(1, self.Kf, self.zf_dim)
            zb_q_cb = zb.view(bs, 1, self.zb_dim) - cb.view(1, self.Kb, self.zb_dim)
            # (B, K)
            log_zf_y = - self.sigma_zf * zf_q_cf.square().sum(dim=-1)
            log_zf_y_alpha = - self.ebm_net.fg_model(zf_q_cf).squeeze(-1)
            post_resp_f = (log_zf_y + log_zf_y_alpha).softmax(dim=-1).detach()

            log_zb_y = - self.sigma_zb * zb_q_cb.square().sum(dim=-1)
            log_zb_y_alpha = - self.ebm_net.bg_model(zb_q_cb).squeeze(-1)
            post_resp_b = (log_zb_y + log_zb_y_alpha).softmax(dim=-1).detach()

            # fg/bg lkhd
            zf_log_lkhd = (post_resp_f * (log_zf_y + log_zf_y_alpha)).sum(dim=-1)
            zb_log_lkhd = (post_resp_b * (log_zb_y + log_zb_y_alpha)).sum(dim=-1)
            
            zs_cat = torch.cat([zs, zb.detach()], dim=1)
            zs_logits = self.ebm_net.sp_model(zs_cat).squeeze(-1)
            zsp_log_lkhd = - zs_logits + \
                           - self.sigma_zs * zs.square().sum(dim=-1)            

            j_log_lkhd = - e_z_log_p.sum() + \
                         - zf_log_lkhd.sum() - zb_log_lkhd.sum() + \
                         - zsp_log_lkhd.sum()

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
                 fg_ma_logits, bg_ma_logits):
        ##### jointly update generator
        self.fg_optimizer.zero_grad()
        self.bg_optimizer.zero_grad()
        self.sp_optimizer.zero_grad()

        # Mutual information
        bs = im_t.size(0)
        cf = self.centroid_f.to(im_t.device).requires_grad_(False)
        cb = self.centroid_b.to(im_t.device).requires_grad_(False)
        # (B, K, D)
        zf_q_cf = zf_logits.view(bs, 1, self.zf_dim) - cf.view(1, self.Kf, self.zf_dim)
        zb_q_cb = zb_logits.view(bs, 1, self.zb_dim) - cb.view(1, self.Kb, self.zb_dim)
        # (B, K)
        with torch.no_grad():
            log_zf_y = - self.sigma_zf * zf_q_cf.square().sum(dim=-1)
            log_zf_y_alpha = - self.ebm_net.fg_model(zf_q_cf).squeeze(-1)
            post_resp_f = (log_zf_y + log_zf_y_alpha).softmax(dim=-1).detach()

            log_zb_y = - self.sigma_zb * zb_q_cb.square().sum(dim=-1)
            log_zb_y_alpha = - self.ebm_net.bg_model(zb_q_cb).squeeze(-1)
            post_resp_b = (log_zb_y + log_zb_y_alpha).softmax(dim=-1).detach()

        # fmi = torch.zeros(1).cuda()        
        fg_logits = self.fge_net(f_logits).view(post_resp_f.size())       
        fmi = dce(post_resp_f, fg_logits).mean()
        # bmi = torch.zeros(1).cuda()
        bg_logits = self.bge_net(b_logits).view(post_resp_b.size())
        bmi = dce(post_resp_b, bg_logits).mean()
        smi = torch.zeros(1).cuda()        

        log_pf = - F.l1_loss(fg_wp, im_t, reduction='none')
        log_pb = - F.l1_loss(bg_wp, im_t, reduction='none')
        with torch.no_grad():
            ga_f = pi_f * log_pf.exp() / (pi_f * log_pf.exp() + pi_b * log_pb.exp() + 1e-8)

        e_z_log_p = ga_f.detach() * ((pi_f + 1e-8).log() + log_pf) \
                  + (1. - ga_f.detach()) * ((pi_b + 1e-8).log() + log_pb)
        recon_loss = -e_z_log_p.mean()        

        G_loss = recon_loss + fmi * 1e-1 + bmi * 1e-1 + smi * 1.

        G_loss.backward()
        self.fg_optimizer.step()
        self.bg_optimizer.step()
        self.sp_optimizer.step()
        # self.tf_optimizer.step()
        # self.tb_optimizer.step()

        return fmi.item(), bmi.item(), smi.item(), recon_loss.item()

    def update_E(self, zp, zn):
        bs = zp.size(0)

        self.ebm_optimizer.zero_grad()        
        
        # posterior resp (K class)
        zpf, zpb, zps = zp[:,:self.zf_dim], \
                        zp[:,self.zf_dim:self.zf_dim + self.zb_dim], \
                        zp[:,-self.zs_dim:]
        cf = self.centroid_f.to(zp.device).requires_grad_(False)
        cb = self.centroid_b.to(zp.device).requires_grad_(False)
        # (B, K, D)
        zf_q_cf = zpf.view(bs, 1, self.zf_dim) - cf.view(1, self.Kf, self.zf_dim)
        zb_q_cb = zpb.view(bs, 1, self.zb_dim) - cb.view(1, self.Kb, self.zb_dim)
        # (B, K)
        log_zf_y = - self.sigma_zf * zf_q_cf.square().sum(dim=-1)
        log_zf_y_alpha = - self.ebm_net.fg_model(zf_q_cf).squeeze(-1)
        post_resp_f = (log_zf_y + log_zf_y_alpha).softmax(dim=-1).detach()

        log_zb_y = - self.sigma_zb * zb_q_cb.square().sum(dim=-1)
        log_zb_y_alpha = - self.ebm_net.bg_model(zb_q_cb).squeeze(-1)
        post_resp_b = (log_zb_y + log_zb_y_alpha).softmax(dim=-1).detach()

        # fg/bg grad w.r.t alpha        
        epf_log_lkhd = - (post_resp_f * log_zf_y_alpha).sum(dim=-1).mean()
        epb_log_lkhd = - (post_resp_b * log_zb_y_alpha).sum(dim=-1).mean()        

        zps_cat = torch.cat([zps, zpb.detach()], dim=1)
        epsp_log_lkhd = self.ebm_net.sp_model(zps_cat).squeeze(-1).mean()

        znf, znb, zns = zn[:,:self.zf_dim], \
                        zn[:,self.zf_dim:self.zf_dim + self.zb_dim], \
                        zn[:,-self.zs_dim:]
        enf_log_lkhd = self.ebm_net.fg_model(znf).squeeze(-1).mean()
        enb_log_lkhd = self.ebm_net.bg_model(znb).squeeze(-1).mean()

        zns_cat = torch.cat([zns, znb.detach()], dim=1)
        ensp_log_lkhd = self.ebm_net.sp_model(zns_cat).squeeze(-1).mean()
        
        loss = (epf_log_lkhd - enf_log_lkhd) + (epb_log_lkhd - enb_log_lkhd) + \
               (epsp_log_lkhd - ensp_log_lkhd)
        loss.backward()

        self.ebm_optimizer.step()

        return loss.item()

    def update_C(self, zp):
        bs = zp.size(0)

        zpf, zpb = zp[:,:self.zf_dim], \
                   zp[:,self.zf_dim:self.zf_dim + self.zb_dim]
        cf = self.centroid_f.to(zp.device).requires_grad_(False)
        cb = self.centroid_b.to(zp.device).requires_grad_(False)
        # (B, K, D)
        zf_q_cf = zpf.view(bs, 1, self.zf_dim) - cf.view(1, self.Kf, self.zf_dim)
        zb_q_cb = zpb.view(bs, 1, self.zb_dim) - cb.view(1, self.Kb, self.zb_dim)
        # (B, K)
        log_zf_y = - self.sigma_zf * zf_q_cf.square().sum(dim=-1)
        log_zf_y_alpha = - self.ebm_net.fg_model(zf_q_cf).squeeze(-1)
        post_resp_f = (log_zf_y + log_zf_y_alpha).softmax(dim=-1).detach()

        log_zb_y = - self.sigma_zb * zb_q_cb.square().sum(dim=-1)
        log_zb_y_alpha = - self.ebm_net.bg_model(zb_q_cb).squeeze(-1)
        post_resp_b = (log_zb_y + log_zb_y_alpha).softmax(dim=-1).detach()

        # estimate mu
        mu_f = post_resp_f.view(bs, self.Kf, 1) * zpf.view(bs, 1, self.zf_dim)
        mu_f_ = torch.div(
                        mu_f.sum(dim=0), 
                        post_resp_f.sum(dim=0).view(self.Kf, 1) + 1e-8
                    )
        mu_b = post_resp_b.view(bs, self.Kb, 1) * zpb.view(bs, 1, self.zb_dim)
        mu_b_ = torch.div(
                        mu_b.sum(dim=0), 
                        post_resp_b.sum(dim=0).view(self.Kb, 1) + 1e-8
                    )

        # EMA
        cf = cf * self.beta + mu_f_ * (1 - self.beta)        
        cb = cb * self.beta + mu_b_ * (1 - self.beta)
        self.centroid_f = cf.cpu()
        self.centroid_b = cb.cpu()

    def learn(self, zp, zn, im_t):
        self.iteration += 1

        im_p, fg_wp, bg_wp, pi_f, fg, bg, pi_b, \
               f_logits, b_logits, s_logits, fg_ma_logits, bg_ma_logits = self(zp, im_t)

        zpf_logits, zpb_logits, zps_logits = zp[:,:self.zf_dim], \
                        zp[:,self.zf_dim:self.zf_dim + self.zb_dim], \
                        zp[:,-self.zs_dim:]
        
        fmi, bmi, smi, l1_loss = self.update_G(
                                im_t, fg_wp, bg_wp, pi_f, pi_b,
                                f_logits, b_logits, s_logits,
                                zpf_logits.detach(), zpb_logits.detach(), zps_logits,
                                fg_ma_logits, bg_ma_logits)

        # update EBM in the latent space
        ebm_loss = self.update_E(zp, zn)        

        with torch.no_grad():
            # update prior cluster
            self.update_C(zp)
            recon_loss = F.mse_loss(im_p, im_t).item()

        logs = [
            # ("l_d_bg", dis_loss_bg),
            # ("l_d_bg_cls", errD_real_bg_cls),
            # ("l_d_all", dis_loss_a),
            # ("post_lkhd", l1_loss),
            ("recon_loss", recon_loss),            
            ("fmi", fmi),
            ("bmi", bmi),
            ("smi", smi),
            ("ebm", ebm_loss)            
        ]
        return logs
