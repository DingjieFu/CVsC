
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class CVsC_model(nn.Module):
    def __init__(self, opt, dataloader):
        super(CVsC_model, self).__init__()
        self.device = opt.device
        self.is_e2e = opt.is_end2end
        # -------------------- Normalization --------------------#
        if opt.dataset == "SUN":
            self.original_att = dataloader.original_att/1.0
        else:
            self.original_att = dataloader.original_att/100.0
        self.att = l2_norm2(self.original_att)
        # ----------------------------------------#
        self.alpha0 = opt.alpha0  # CE
        self.alpha1 = opt.alpha1  # calibrate
        self.alpha2 = opt.alpha2  # CCL
        self.alpha3 = opt.alpha3  # SCI
        self.alpha4 = opt.alpha4  # TCI

        self.output_dim = self.att.size(-1)
        self.nclass = self.att.size(0)
        self.seenclass = dataloader.seenclasses
        self.unseenclass = dataloader.unseenclasses
        self.is_bias = opt.is_bias
        bias = opt.bias

        if opt.is_bias:
            self.bias = nn.Parameter(torch.tensor(bias), requires_grad=False)
            mask_bias = np.ones((1, self.nclass))
            mask_bias[:, self.seenclass.cpu().numpy()] *= 0
            self.mask_bias = nn.Parameter(torch.tensor(
                mask_bias).float(), requires_grad=False)
            self.vec_bias = (self.mask_bias*self.bias).to(self.device)

        if opt.w2v_att == True:
            self.init_w2v_att = F.normalize(dataloader.w2v_att.clone().detach())
            self.V = nn.Parameter(self.init_w2v_att.clone(), requires_grad=opt.trainable_w2v)
        else:
            self.V = nn.Parameter(nn.init.normal_(torch.empty(
                dataloader.w2v_att.size(0), dataloader.w2v_att.size(1))), requires_grad=True)
        self.normalize_V = True  # whether normalize self.V

        if opt.backbone.lower() == 'vit':
            self.dim_f = 768
        else:
            self.dim_f = 2048
        self.dim_v = 300
        self.W1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
 

    def forward(self, img):
        vv = self.compute_V()
        if self.dim_f == 768:
            img = img[:, 1:, :]
            img = torch.permute(img, (0, 2, 1))
        else:
            shape = img.shape
            img = img.reshape(shape[0], shape[1], shape[2]*shape[3])
        av = F.normalize(img, dim=1)
        A = torch.einsum('iv,vf,bfr->bir', vv, self.W1, av)
        A_ = -1.0 * A
        A_ = nn.Softmax(dim=-1)(A_)
        A = nn.Softmax(dim=-1)(A)

        F_p = torch.einsum('bir,bfr->bif', A, av)
        F_p_ = F_p[torch.randperm(F_p.size(0)), :, :]
        F_p_ = F_p_[:, torch.randperm(F_p.size(1)), :]
        F_p_ = F_p_[:, :, torch.randperm(F_p.size(2))]

        Pred_att = torch.einsum('iv,vf,bif->bi', vv, self.W2, F_p)
        Pred_att_ = torch.einsum('iv,vf,bif->bi', vv, self.W2, F_p_)

        S_pp = torch.einsum('ki,bi->bik', self.att, Pred_att)
        S_pp = torch.sum(S_pp, dim=1)
        S_pp_ = torch.einsum('ki,bi->bik', self.att, Pred_att_)
        S_pp_ = torch.sum(S_pp_, dim=1)

        if self.is_bias:
            S_pp = S_pp + self.vec_bias
        """
            F_p -> (bs, attribute_dim, visual_dim)
            Pred_att -> (bs, attribute_dim)
            S_pp -> (bs, class_num)
            F_p_, Pred_att_, S_pp_ -> 
        """
        package = {'A': A, 'F_p': F_p, 'Pred_att': Pred_att, 'S_pp': S_pp, 'F_p_': F_p_,
                   'Pred_att_': Pred_att_, 'S_pp_': S_pp_}
        return package

    def compute_loss(self, in_package):
        # loss self-calibration
        loss_cal = self.compute_loss_Self_Calibrate(in_package)
        if self.is_bias:
            in_package['S_pp'] = in_package['S_pp'] - self.vec_bias
        # loss CE
        loss_CE = nn.CrossEntropyLoss()(
            in_package['S_pp'], in_package['batch_label'])

        norm = nn.BatchNorm1d(self.original_att.shape[-1], device=self.device)
        orig_Ground_att = self.original_att[in_package['batch_label'].type(torch.long)]
        if self.original_att[in_package['batch_label']].size(0) == 1:
            Ground_att =  self.original_att[in_package['batch_label'].type(torch.long)]
        else:
            Ground_att = norm(self.original_att[in_package['batch_label'].type(torch.long)])
        orig_Pred_att = in_package['Pred_att']
        orig_Pred_att_ = in_package['Pred_att_']
        Pred_att_all = torch.concat([in_package['Pred_att'], in_package['Pred_att_']])
        Pred_att = norm(Pred_att_all)[:orig_Pred_att.shape[0], :]
        Pred_att_ = norm(Pred_att_all)[-orig_Pred_att_.shape[0]:, :]

        # CCL
        dif_pg = Ground_att - Pred_att
        loss_CCL = ((dif_pg*(dif_pg < 0).float())**2).sum()/((dif_pg < 0).float().sum()+1) +\
            ((dif_pg*(dif_pg > 0).float())**2).sum()/((dif_pg > 0).float().sum()+1) +\
            abs(orig_Ground_att.var()-orig_Pred_att.var())

        # SCI
        dif_pp_ = Ground_att - Pred_att + Pred_att_
        loss_SCI = dif_pp_.abs().mean()

        # TCI
        S_pp_diff = in_package['S_pp']-in_package['S_pp_']
        loss_TCI = nn.CrossEntropyLoss()(S_pp_diff, in_package['batch_label'])

        # total loss
        loss = self.alpha0 * loss_CE + self.alpha1 * loss_cal + self.alpha2 * \
            loss_CCL + self.alpha3 * loss_SCI + self.alpha4 * loss_TCI

        out_package = {'loss': loss, 'loss_CE': loss_CE, 'loss_Cali': loss_cal, 'loss_CCL': loss_CCL,
                       'loss_SCI': loss_SCI, 'loss_TCI': loss_TCI}

        return out_package

    def compute_V(self):
        if self.normalize_V:
            V_n = F.normalize(self.V, dim=1)
        else:
            V_n = self.V
        return V_n

    def compute_loss_Self_Calibrate(self, in_package):
        """
            - Lcal({s})
        """
        S_pp = in_package['S_pp']
        Prob_all = F.softmax(S_pp, dim=-1)
        Prob_unseen = Prob_all[:, self.unseenclass]
        assert Prob_unseen.size(1) == len(self.unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp


def l2_norm2(input):
    norm = torch.norm(input, 2, -1, True)
    output = torch.div(input, norm)
    return output
