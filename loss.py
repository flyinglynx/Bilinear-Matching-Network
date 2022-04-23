import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    def __init__(self, device='cpu', reduction='mean', downsampling_rate=32):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.downsampling_rate = downsampling_rate
        self.kernel = torch.ones(1, 1, downsampling_rate, downsampling_rate).to(device)

    def forward(self, outputs, density_map):
        if self.downsampling_rate > 1:
            density_map = F.conv2d(density_map, self.kernel, stride=self.downsampling_rate)
        loss = F.l1_loss(outputs, density_map, reduction=self.reduction)
        return loss


class L2Loss(nn.Module):
    def __init__(self, device='cpu', reduction='mean', downsampling_rate=32):
        super(L2Loss, self).__init__()
        self.reduction = reduction
        self.downsampling_rate = downsampling_rate
        self.kernel = torch.ones(1, 1, downsampling_rate, downsampling_rate).to(device)

    def forward(self, outputs, density_map):
        if self.downsampling_rate > 1:
            density_map = F.conv2d(density_map, self.kernel, stride=self.downsampling_rate)
        loss = F.mse_loss(outputs, density_map, reduction=self.reduction)
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, device='cpu', downsampling_rate=32):
        super().__init__()
        self.downsampling_rate = downsampling_rate
        self.kernel = torch.ones(1, 1, downsampling_rate, downsampling_rate).to(device)
        
    def forward(self, corr_map, pt_map, density_map):
        # resize the pt_map to the shape of features
        pt_map = F.conv2d(pt_map.float(), self.kernel, stride=self.downsampling_rate).bool()
        bs, _, h, w = pt_map.shape
        pt_map = pt_map.flatten(2).view(bs, h*w)
        
        # corr_map: shape of B * HW * query_number
        corr = torch.exp(corr_map)
        corr = corr.mean(dim=-1, keepdim=False) # shape of B * HW
        
        loss = 0
        for idx in range(bs):
            pos_corr = corr[idx][pt_map[idx]].sum()
            neg_corr = corr[idx][~pt_map[idx]].sum()
            sample_loss = -1 * torch.log(pos_corr / (neg_corr + pos_corr + 1e-10))
            loss += sample_loss
            
        return loss / bs
    
class TrainingLoss(nn.Module):
    def __init__(self, counting_loss, contrast_loss=None, wcl=1e-5):
        super().__init__()
        self.counting_loss = counting_loss
        self.contrast_loss = contrast_loss
        self.wcl = wcl
        
    def forward(self, outputs, density_map, pt_map):
        dest, corr_map = outputs['density_map'], outputs['corr_map']
        counting_loss = self.counting_loss(dest, density_map)
        if self.contrast_loss:
            contrast_loss = self.contrast_loss(corr_map, pt_map, density_map) * self.wcl
            return counting_loss, contrast_loss
        else:
            return counting_loss, 0
    
def get_loss(cfg):
    # get counting loss
    if cfg.TRAIN.counting_loss == 'l1loss':
        counting_loss = L1Loss(device=cfg.TRAIN.device, reduction='mean', downsampling_rate=cfg.DATASET.downsampling_rate)
    elif cfg.TRAIN.counting_loss == 'l2loss':
        counting_loss = L2Loss(device=cfg.TRAIN.device, reduction='mean', downsampling_rate=cfg.DATASET.downsampling_rate)
    else:
        raise NotImplementedError
    
    feature_drate = 32 if cfg.MODEL.backbone_layer == "layer4" else 16
    if cfg.TRAIN.contrast_loss == 'info_nce':
        contrast_loss = InfoNCELoss(device=cfg.TRAIN.device, downsampling_rate=feature_drate)
    elif cfg.TRAIN.contrast_loss == 'none':
        contrast_loss = None
    else:
        raise NotImplementedError
    
    loss = TrainingLoss(counting_loss=counting_loss,
                        contrast_loss=contrast_loss,
                        wcl = cfg.TRAIN.contrast_weight)
    
    return loss

if __name__ == '__main__':
    pass
    
    