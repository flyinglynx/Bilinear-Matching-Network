"""
Feature matcher class for Class Agnostic Object counting.
"""
import torch
from torch import nn

# Measure similarity with fixed inner product 
class InnerProductMatcher(nn.Module):
    def __init__(self, pool='mean'):
        super().__init__()
        self.pool = pool
    
    def forward(self, features, patches_feat):
        bs, c, h, w = features.shape
        proj_feat = features.flatten(2).permute(0, 2, 1)  # bs * hw * c
        patches_feat = patches_feat.permute(1, 2, 0)      # bs * c * exemplar_number
        energy = torch.bmm(proj_feat, patches_feat)       # bs * hw * exemplar_number
        
        if self.pool == 'mean':
            corr = energy.mean(dim=-1, keepdim=True)
        elif self.pool == 'max':
            corr = energy.max(dim=-1, keepdim=True)[0]
        out = torch.cat((features, corr), dim=-1) # bs * hw * dim

        return out.permute(0, 2, 1).view(bs, c+1, h, w), energy
    

# Measure similarity with bilinear similarity metric
class BilinearSimilarityMatcher(nn.Module):
    def __init__(self, hidden_dim, proj_dim, activation='relu', pool='mean', use_bias=False):
        super().__init__()
        self.pool = pool
        self.query_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self.key_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self._weight_init_()
    
    def forward(self, features, patches):
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(0, 2, 1) # bs * hw * c
        proj_feat = self.query_conv(features)
        patches_feat = self.key_conv(patches)

        patches_feat = patches_feat.permute(1, 2, 0)    # bs * c * exemplar_number
        energy = torch.bmm(proj_feat, patches_feat)     # bs * hw * exemplar_number
        
        if self.pool == 'mean':
            corr = energy.mean(dim=-1, keepdim=True)
        elif self.pool == 'max':
            corr = energy.max(dim=-1, keepdim=True)[0]
            
        out = torch.cat((features, corr), dim=-1)       # bs * hw * (c+1)

        return out.permute(0, 2, 1).view(bs, c+1, h, w), energy 
    
    def _weight_init_(self):
        for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

class DynamicSimilarityMatcher(nn.Module):
    def __init__(self, hidden_dim, proj_dim, dynamic_proj_dim, activation='tanh', pool='mean', use_bias=False):
        super().__init__()
        self.query_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self.key_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self.dynamic_pattern_conv = nn.Sequential(nn.Linear(in_features=proj_dim, out_features=dynamic_proj_dim),
                                          nn.ReLU(),
                                          nn.Linear(in_features=dynamic_proj_dim, out_features=proj_dim))
        
        self.softmax  = nn.Softmax(dim=-1)
        self._weight_init_()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise NotImplementedError
            
    def forward(self, features, patches):
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # hw * bs * dim
        
        proj_feat = self.query_conv(features)
        patches_feat = self.key_conv(patches)
        patches_ca = self.activation(self.dynamic_pattern_conv(patches_feat))
        
        proj_feat = proj_feat.permute(1, 0, 2)
        patches_feat = (patches_feat * (patches_ca + 1)).permute(1, 2, 0)  # bs * c * exemplar_number        
        energy = torch.bmm(proj_feat, patches_feat)                        # bs * hw * exemplar_number

        corr = energy.mean(dim=-1, keepdim=True)
        out = features.permute(1,0,2)  # hw * bs * c
        out = torch.cat((out, corr), dim=-1)
        
        out = out.permute(1,0,2)
        return out.permute(1, 2, 0).view(bs, c+1, h, w), energy 
    
    def _weight_init_(self):
        for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def build_matcher(cfg):
    matcher_name = cfg.MODEL.matcher
    if matcher_name == 'inner_product_matcher':
        return InnerProductMatcher()
    elif matcher_name == 'bilinear_similarity_matcher':
        return BilinearSimilarityMatcher(hidden_dim=cfg.MODEL.hidden_dim,
                                         proj_dim=cfg.MODEL.matcher_proj_dim,
                                         use_bias=cfg.MODEL.use_bias)
    elif matcher_name == 'dynamic_similarity_matcher':
        return DynamicSimilarityMatcher(hidden_dim=cfg.MODEL.hidden_dim,
                                        proj_dim=cfg.MODEL.matcher_proj_dim,
                                        dynamic_proj_dim=cfg.MODEL.dynamic_proj_dim,
                                        use_bias=cfg.MODEL.use_bias)
    else:
        raise NotImplementedError



