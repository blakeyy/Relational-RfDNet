import torch
import torch.nn as nn
import numpy as np
from models.registers import MODULES
 
@MODULES.register_module
class RelationModule(nn.Module):
    def __init__(self, cfg, optim_spec = None):
        super(RelationModule, self).__init__()
        self.cfg = cfg
        self.optim_spec = optim_spec
        appearance_feature_dim = cfg.config['model']['enhance_recognition']['appearance_feature_dim']
        key_feature_dim = cfg.config['model']['enhance_recognition']['key_feature_dim']
        geo_feature_dim = cfg.config['model']['enhance_recognition']['geo_feature_dim']
        self.isDuplication = cfg.config['model']['enhance_recognition']['isDuplication']
        self.Nr = cfg.config['model']['enhance_recognition']['n_relations']
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))
    def forward(self, input_data ):
        if(self.isDuplication):
            f_a, embedding_f_a, position_embedding =input_data
        else:
            f_a, position_embedding = input_data # f_a: (B,K,appearance_feature_dim), position_embedding: (B,K,K,dim_g)
        isFirst=True
        for N in range(self.Nr):
            if(isFirst):
                if(self.isDuplication):
                    concat = self.relation[N](embedding_f_a,position_embedding)  #(B,K,dim_k)
                else:
                    concat = self.relation[N](f_a,position_embedding)
                isFirst=False
            else:
                if(self.isDuplication):
                    concat = torch.cat((concat, self.relation[N](embedding_f_a, position_embedding)), -1)
                else:
                    concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
        return concat+f_a
class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=768,key_feature_dim = 96, geo_feature_dim = 96):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, f_a, position_embedding):#f_a: (B,K,appearance_feature_dim), position_embedding: (B,K,K,dim_g)
        B,K,_ = f_a.size()

        w_g = self.relu(self.WG(position_embedding)) # (B,K,K,1)

        w_k = self.WK(f_a) # (B,K,dim_k)
        w_k = w_k.view(B,K,1,self.dim_k)

        w_q = self.WQ(f_a) # (B,K,dim_k)
        w_q = w_q.view(B,1,K,self.dim_k)

        scaled_dot = torch.sum((w_k*w_q),-1 ) # (B,K,K). Note that 1st K is key, 2nd K is query 
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(B,K,K) # Note that 1st K is key, 2nd K is query 
        w_a = scaled_dot.view(B,K,K) 

        w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a # (B,K,K)
        w_mn = torch.nn.Softmax(dim=1)(w_mn) # compute softmax along key dimension 

        w_v = self.WV(f_a) # (B,K,dim_k)

        w_mn = w_mn.view(B,K,K,1) # (B,K,K,1)
        w_v = w_v.view(B,K,1,-1) # (B,K,1,dim_k)
        output = w_mn*w_v # (B,K,K, dim_k)
        output = torch.sum(output,1) # (B,K,dim_k)

        return output