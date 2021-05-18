import torch
import torch.nn as nn
import numpy as np

from net_utils.nn_distance import nn_distance

from net_utils.relation_tool import PositionalEmbedding

from models.registers import MODULES
from models.iscnet.modules.proposal_module import decode_scores

 
@MODULES.register_module
class RelationalProposalModule(nn.Module):
    def __init__(self, cfg, optim_spec = None):
        '''
        Relation-based Proposal Module to enhance detected proposals.
        :param config: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(RelationalProposalModule, self).__init__()
        
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        
        '''Parameters'''
        self.num_class = cfg.dataset_config.num_class
        self.num_heading_bin = cfg.dataset_config.num_heading_bin
        self.num_size_cluster = cfg.dataset_config.num_size_cluster

        appearance_feature_dim = cfg.config['model']['enhance_recognition']['appearance_feature_dim']
        key_feature_dim = cfg.config['model']['enhance_recognition']['key_feature_dim']
        geo_feature_dim = cfg.config['model']['enhance_recognition']['geo_feature_dim']
        self.isDuplication = cfg.config['model']['enhance_recognition']['isDuplication']
        self.Nr = cfg.config['model']['enhance_recognition']['n_relations']
        self.dim_g = geo_feature_dim

        '''Modules'''
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))

        self.feature_transform1 = nn.Sequential(nn.Conv1d(128,128,1), \
                                            nn.BatchNorm1d(128), \
                                            nn.ReLU(), \
                                            nn.Conv1d(128, self.cfg.config['model']['enhance_recognition']['appearance_feature_dim'], 1))
        
        self.feature_transform2 = nn.Sequential(nn.Conv1d(self.cfg.config['model']['enhance_recognition']['appearance_feature_dim'],128,1), \
                                            nn.BatchNorm1d(128), \
                                            nn.ReLU(), \
                                            nn.Conv1d(128, 128, 1))

        self.proposal_generation = nn.Sequential(nn.Conv1d(128,128,1), \
                                            nn.BatchNorm1d(128), \
                                            nn.ReLU(), \
                                            nn.Conv1d(128,128,1), \
                                            nn.BatchNorm1d(128), \
                                            nn.ReLU(), \
                                            nn.Conv1d(128,5 + self.num_heading_bin*2 + self.num_size_cluster*4 + self.num_class,1))
        

    def forward(self, proposal_features, end_points, data):
        center = end_points['center'] # (B, K, 3)
        size_scores = end_points['size_scores'] # (B, K, num_size_cluster)
    
        # choose the cluster for each proposal based on GT class of that proposal. GT class of each proposal is the closest GT box to each predicted proposal
        aggregated_vote_xyz = end_points['aggregated_vote_xyz'] #(B,K,3)
        gt_center = data['center_label'][:,:,0:3]
        _, ind1, _, _ = nn_distance(aggregated_vote_xyz, gt_center)
        object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1
        size_class_label = torch.gather(data['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2), object_assignment: (B,K) with values in 0,1,...,K2-1
        size_label_one_hot = torch.cuda.FloatTensor(size_scores.shape).zero_()
        size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
        size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3)#(B, K, num_size_cluster, 3)
        size_residual_label = torch.gather(data['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)

        # get predicted l,w,h from residual
        # residual = (h-h')/h', h: predicted height, h': mean height of corresponding class 
        mean_size_arr = self.cfg.dataset_config.mean_size_arr
        mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3)
        mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B, K,3)
        gt_size = size_residual_label + mean_size_label # (B,K,3)

        # get geometric feature and feed it into PositionalEmbedding 
        geometric_feature = torch.cat([center, gt_size], dim=-1) # (B, K, 6)  ### Ingo: Is the heading direction inlcuded here? ###
        position_embedding = PositionalEmbedding(geometric_feature) # (B,K,K, dim_g)

        #transform proposal_features from 128-dim to appearance_feature_dim 
        proposal_features = self.feature_transform1(proposal_features)  #(B,appearance_feature_dim, K)
        proposal_features = proposal_features.transpose(1, 2).contiguous() # (B, K, appearance_feature_dim)

        # proposal_features: (B,K,appearance_feature_dim)
        # positional_embedding: (B,K,K,dim_g)
        if(self.isDuplication):
            f_a, embedding_f_a, position_embedding = (proposal_features, position_embedding)
        else:
            f_a, position_embedding = (proposal_features, position_embedding) #input_data # f_a: (B,K,appearance_feature_dim), position_embedding: (B,K,K,dim_g)
        
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
        
        ### TODO: Add learning parameter to let the network decide during training if relation module is used ####
        proposal_features = concat+f_a      # proposal_features: (B,K, appearance_feature_dim)

        proposal_features = proposal_features.transpose(1,2).contiguous() #(B,appearance_feature_dim, K)
        proposal_features = self.feature_transform2(proposal_features) # (B,128,K)

        net = self.proposal_generation(proposal_features) # # (B, 2+3+num_heading_bin*2+num_size_cluster*4 + num_class, K)
        end_points = decode_scores(net, end_points, self.num_heading_bin, self.num_size_cluster)
        
        return end_points, proposal_features

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