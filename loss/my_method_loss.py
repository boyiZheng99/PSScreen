import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class partial_BCELoss(nn.Module):  #For cross entropy loss for known classes and pseudo label consistency loss

    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(partial_BCELoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.reduce = reduce
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        input, target = input.float(), target.float()

        positive_mask = (target > self.margin).float()    #Record the positions of positive classes.
        negative_mask = (target < -self.margin).float()   #Record the positions of negaive classes.

        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)

        positive_count = torch.sum(positive_mask, dim=0)  
        negative_count = torch.sum(negative_mask, dim=0)  
        total_count = positive_count + negative_count  

        epsilon = 1e-8
        positive_weight = negative_count / (total_count + epsilon)  
        negative_weight = positive_count / (total_count + epsilon)  

        pos_weight_mask = torch.ones_like(positive_weight)
        neg_weight_mask = torch.ones_like(negative_weight)

        pos_weight_mask[0] = positive_weight[0]
        neg_weight_mask[0] = negative_weight[0]  


        weighted_positive_loss = pos_weight_mask * positive_mask * positive_loss  #Only compute loss on positive classes.
        weighted_negative_loss = neg_weight_mask * negative_mask * negative_loss  #Only compute loss on negative classes.


        loss = weighted_positive_loss + weighted_negative_loss  

        if self.reduce:
            if self.size_average:
                active_samples = (positive_mask + negative_mask) > 0
                return torch.mean(loss[active_samples]) if torch.sum(active_samples) > 0 else torch.mean(loss)
            else:
                active_samples = (positive_mask + negative_mask) > 0
                return torch.sum(loss[active_samples]) if torch.sum(active_samples) > 0 else torch.sum(loss)

        return loss


class Partial_MultiLabelKLDivergenceLoss(nn.Module):  #Compute the self-distillation loss
   
    def __init__(self, reduction='mean', eps=1e-10):
        super(Partial_MultiLabelKLDivergenceLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_logits, target_logits, real_label):
      

        pred_probs = torch.sigmoid(pred_logits)
        target_probs = torch.sigmoid(target_logits)

        mask = (real_label!=0).float()  #Record the positions of labeled classes.


        kl_loss = mask * (target_probs * (torch.log(target_probs + self.eps) - torch.log(pred_probs + self.eps)) +
                          (1 - target_probs) * (torch.log(1 - target_probs + self.eps) - torch.log(1 - pred_probs + self.eps)))  #Only compute KL loss on labeled classes.


        if self.reduction == 'mean':
            kl_loss = kl_loss.sum() / (mask.sum()+1e-8)
        elif self.reduction == 'sum':
            kl_loss = kl_loss.sum()
        elif self.reduction == 'none':
            pass 

        return kl_loss


class MMDLoss(nn.Module):  #Compute the feature-distillation loss.
  
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        B, C, D = source.shape
        mmd_loss_list = []
        
        for c in range(C):     #Compute the channel-wise MMD loss.
            source_c = source[:, c, :]  
            target_c = target[:, c, :]  
            
            if self.kernel_type == 'linear':
                mmd_loss_c = self.linear_mmd2(source_c, target_c)
            elif self.kernel_type == 'rbf':
                batch_size = int(source_c.size()[0])
                kernels = self.guassian_kernel(
                    source_c, target_c, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                mmd_loss_c = torch.mean(XX + YY - XY - YX)
            
            mmd_loss_list.append(mmd_loss_c)
        
        total_loss = torch.mean(torch.stack(mmd_loss_list))
        return total_loss



