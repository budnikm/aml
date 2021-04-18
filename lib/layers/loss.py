import torch
import torch.nn as nn
import numpy as np
import lib.layers.functional as LF
import torch.nn.functional as F
import pdb



# --------------------------------------
# Loss/Error layers
# --------------------------------------

class ContrastiveLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, x, label):
        return LF.contrastive_loss(x, label, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'



class ContrastiveDistLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(ContrastiveDistLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, x, label, dist):
        return LF.contrastive_loss_dist(x, label, dist, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


class TripletLoss(nn.Module):

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x, label):
        return LF.triplet_loss(x, label, margin=self.margin)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


class CrossEntropyLoss(nn.Module):

    def __init__(self, temp=1, eps=1e-6):
        super(CrossEntropyLoss, self).__init__()
        self.temp = temp
        self.eps = eps
        

    def forward(self, x, label):
        return LF.cross_entropy_loss(x, label, temp=self.temp, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'temp=' + '{:.2f}'.format(self.temp) + ')'



class CrossEntropyDistLoss(nn.Module):

    def __init__(self, temp=1, eps=1e-6):
        super(CrossEntropyDistLoss, self).__init__()
        self.temp = temp
        self.eps = eps

    def forward(self, x, label):
        return LF.cross_entropy_loss_dist(x, label, temp=self.temp, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'temp=' + '{:.2f}'.format(self.temp) + ')'



class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0 
        self.scale_neg = 40.0 

    def forward(self, feats, labels):

        assert feats.size(1) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(1)
        sim_mat = torch.matmul(torch.t(feats), feats)

        epsilon = 1e-5
        
        pos_pair = sim_mat[0][labels == 1]
        
        neg_pair = sim_mat[0][labels == 0]

        
        loss = 0
        if len(neg_pair) >= 1 or len(pos_pair) >= 1:
        # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            
            loss = pos_loss + neg_loss
        if loss == 0:
            return torch.zeros([], requires_grad=True)
        
        return loss

class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * LF.pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * LF.pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss

class RKdAD(nn.Module):
    def __init__(self):
        super(RKdAD, self).__init__()

    def pdist(self, e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res 

    def forward(self, student, teacher):
        # N x C
        # N x N x C
        
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        d = self.pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss_a = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        loss_d = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')


        return 2*loss_a+loss_d

