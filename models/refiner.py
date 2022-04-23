"""
Feature refiner class for Class Agnostic Object counting.
"""
import torch
from torch import nn
import copy 


'''
Self similarity module
'''  
class SelfSimilarityModule(nn.Module):
    def __init__(self, hidden_dim, proj_dim, layer_number):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(SelfSimilarityLayer(hidden_dim=hidden_dim, proj_dim=proj_dim)) for i in range(layer_number)])
    def forward(self, features, patches):
        for layer in self.layers:
            features, patches = layer(features, patches)
        return features, patches

'''
Layer in self similarity module
'''
class SelfSimilarityLayer(nn.Module):
    def __init__(self, hidden_dim, proj_dim, dropout_rate=0.0):
        super().__init__()
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        
        self.query_conv = nn.Linear(hidden_dim, proj_dim)
        self.key_conv = nn.Linear(hidden_dim, proj_dim)
        self.value_conv = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.post_conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                         nn.ReLU())
        
        self.softmax  = nn.Softmax(dim=-1)
            
        self._weight_init_()
    
    def forward(self, features, patches):
        """
            inputs :
                x : input feature maps (B X C X W X H)
                patches: feature vectors of exemplar patches (query_number X B X C)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = features.size()
        query_number = patches.shape[0]
        features = features.view(m_batchsize, -1, width*height).permute(0, 2, 1) # B X N X C
        appended_features = torch.cat((features, patches.permute(1, 0, 2)), dim=1) # B X (N+query_number) X C
        proj_value = self.value_conv(features).view(m_batchsize,-1,width*height) # B X C X (N+query_number)
        
        proj_query = self.query_conv(appended_features)
        proj_key = self.key_conv(appended_features).permute(0, 2, 1) # B X C X (N + query_number)
        proj_value = self.value_conv(appended_features) # B X (N+query_number) X C
        
        energy =  torch.bmm(proj_query, proj_key) # B X (N+query_number) X (N+query_number)
        attention = self.softmax(energy) 

        out = torch.bmm(proj_value.permute(0, 2, 1), attention.permute(0,2,1)) # B X C X (N+query_number)
        out = self.gamma * self.dropout(out) + appended_features.permute(0,2,1)
        #out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1) # shape of B X (N+query_number) X dim
        #self.out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1) # shape of B X (N+query_number) X dim
        
        out_feat, out_patch = out[:, :, :-1*query_number], out[:,:,-1*query_number:]
        out_feat = out_feat.reshape(m_batchsize, C, width, height) # B X C X H X W
        out_patch = out_patch.permute(2, 0, 1) # query_number * B * dim
        
        return self.post_conv(out_feat), out_patch
    
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


class NoneRefiner(object):
    def __init__(self):
        pass
    
    def __call__(self, features, patches):
        return features, patches


def build_refiner(cfg):
    refiner_name = cfg.MODEL.refiner 
    #print(refiner_name)
    
    if refiner_name == 'self_similarity_module':
        return SelfSimilarityModule(hidden_dim=cfg.MODEL.hidden_dim,
                                    proj_dim=cfg.MODEL.refiner_proj_dim,
                                    layer_number=cfg.MODEL.refiner_layers)
    elif refiner_name == 'none':
        return NoneRefiner()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    refiner = SelfSimilarityModule(hidden_dim=256,
                                    proj_dim=128,
                                    layer_number=1)
    
    imgF = torch.rand(5, 256, 12, 12)
    patchF = torch.rand(3, 5, 256)
    
    feature, patch = refiner(imgF, patchF)

