import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from net_utils.nn_distance import nn_distance

from net_utils.relation_tool import PositionalEmbedding

from models.registers import MODULES
from models.iscnet.modules.proposal_module import decode_scores

from configs.scannet_config import ScannetConfig #param2obb
from net_utils.ap_helper import parse_predictions

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
        #self.gamma = nn.Parameter(torch.ones(1)) # requires_grad is True by default for Parameter
        #nn.init.constant_(self.gamma, 0.0)

        #self.feature_transform1 = nn.Sequential(nn.Conv1d(128,128,1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128,128,1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128, appearance_feature_dim, 1)) 

        self.pos_embedding = PositionEmbeddingLearned(6, geo_feature_dim)

        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim=key_feature_dim, geo_feature_dim=geo_feature_dim))

        ##### Adding concat to f_a
        self.feature_transform2 = nn.Sequential(nn.Conv1d(appearance_feature_dim, 128, 1), \
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
        
        ##### Concat concat to f_a
        #self.feature_transform2 = nn.Sequential(nn.Conv1d(appearance_feature_dim + self.dim_g*self.Nr, 128, 1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128, 128, 1))
        #self.proposal_generation = nn.Sequential(nn.Conv1d(128,128,1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128,128,1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128, 5 + self.num_heading_bin*2 + self.num_size_cluster*4 + self.num_class, 1))
        
        #self.feature_transform1 = nn.Sequential(nn.Conv1d(128,128,1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128, appearance_feature_dim, 1))
        
        #self.feature_transform2 = nn.Sequential(nn.Conv1d(appearance_feature_dim, 128, 1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128, 128, 1))


        #self.proposal_generation = nn.Sequential(nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128,128,1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128,128,1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128, 5 + self.num_heading_bin*2 + self.num_size_cluster*4 + self.num_class, 1))
        
        # Prediction Head
        #self.prediction_heads = nn.ModuleList()
        #for i in range(self.num_decoder_layers):
        #    self.prediction_heads.append(nn.Sequential(nn.Conv1d(128,128,1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128,128,1), \
        #                                    nn.BatchNorm1d(128), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(128, 5 + self.num_heading_bin*2 + self.num_size_cluster*4 + self.num_class, 1)))

        
        #self.init_weights()
        #self.bn_momentum = cfg.config['bnscheduler']['bn_momentum_init']
        #self.init_bn_momentum()
        #self.relation.apply(init_weights)
        #self.feature_transform1.apply(init_weights)
        #self.feature_transform2.apply(init_weights)
        #self.proposal_generation.apply(init_weights)

    def forward(self, proposal_features, end_points, data, mode='train'):
        center = end_points['center'] # (B, K, 3)
        #size_scores = end_points['proposal_size_scores'] # (B, K, num_size_cluster)

        ### Heading not used yet
        #heading_scores = end_points['proposal_heading_scores']  # Bxnum_proposalxnum_heading_bin
        #heading_residuals_normalized = end_points['proposal_heading_residuals_normalized']  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        #heading_class = 
        #heading_residual =
        
        #######
        config_dict = self.cfg.eval_config

        pred_size_class = torch.argmax(end_points['size_scores'], -1)  # B,num_proposal
        size_residuals = end_points['size_residuals_normalized'] * torch.from_numpy(
            config_dict['dataset_config'].mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        
        pred_size_residual = torch.gather(size_residuals, 2,
                                        pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                            3))  # B,num_proposal,1,3
        pred_size_residual.squeeze_(2)

        mean_size_arr = torch.from_numpy(config_dict['dataset_config'].mean_size_arr.astype(np.float32)).cuda()
        pred_size_class = torch.squeeze(pred_size_class.type(torch.cuda.LongTensor)) ## Problem if batch_size==1 -> change where to squeeze
        temp = mean_size_arr[pred_size_class, :]
        box_size = temp + pred_size_residual

        #size_class_label = torch.gather(data['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2), object_assignment: (B,K) with values in 0,1,...,K2-1
        #size_label_one_hot = torch.cuda.FloatTensor(size_scores.shape).zero_()
        #size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
        #size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3)#(B, K, num_size_cluster, 3)
        #size_residual_label = torch.gather(data['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)

        # get predicted l,w,h from residual
        # residual = (h-h')/h', h: predicted height, h': mean height of corresponding class 
        #mean_size_arr = self.cfg.dataset_config.mean_size_arr
        #mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3)
        #mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B, K,3)
        #gt_size = size_residual_label + mean_size_label # (B,K,3)

        #print("Mode relation model forward(): " + str(self.cfg.config['mode']))
        # choose the cluster for each proposal based on GT class of that proposal. GT class of each proposal is the closest GT box to each predicted proposal
        #aggregated_vote_xyz = end_points['aggregated_vote_xyz'] #(B,K,3)
        #gt_center = data['center_label'] #(B,K2,3)
        #_, ind1, _, _ = nn_distance(aggregated_vote_xyz, gt_center)
        #object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1     

        #size_class_label = torch.gather(data['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2), object_assignment: (B,K) with values in 0,1,...,K2-1
        #size_residual_label = torch.gather(data['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3))    # select (B,K,3) from (B,K2,3)
        #mean_size_label = torch.from_numpy(self.cfg.dataset_config.mean_size_arr.astype(np.float32)).to('cuda')[size_class_label]    # (B,K,3)
        #gt_size = size_residual_label + mean_size_label # (B,K,3)

        #if mode == 'train':
        # choose the cluster for each proposal based on GT class of that proposal. GT class of each proposal is the closest GT box to each predicted proposal
        #aggregated_vote_xyz = end_points['aggregated_vote_xyz'] #(B,K,3)
        #gt_center = data['center_label'][:,:,0:3]
        #_, ind1, _, _ = nn_distance(aggregated_vote_xyz, gt_center)
        #object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1
        #size_class_label = torch.gather(data['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2), object_assignment: (B,K) with values in 0,1,...,K2-1
        #size_label_one_hot = torch.cuda.FloatTensor(size_scores.shape).zero_()
        #size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
        #size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3)#(B, K, num_size_cluster, 3)
        #size_residual_label = torch.gather(data['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)

        # get predicted l,w,h from residual
        # residual = (h-h')/h', h: predicted height, h': mean height of corresponding class 
        #mean_size_arr = self.cfg.dataset_config.mean_size_arr
        #mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3)
        #mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B, K,3)
        #gt_size = size_residual_label + mean_size_label # (B,K,3)
        #box_size = size_residual_label + mean_size_label # (B,K,3)
        #else:       ## for val and test?
            ## use predicted gt_size
        #    pass

        # get geometric feature and feed it into PositionalEmbedding          ### TODO: Only use GT during train mode 
        geometric_feature = torch.cat([center, box_size], dim=-1) # (B, K, 6)  ### Ingo: Is the heading direction inlcuded here? ###
        #geometric_feature = torch.cat([center, gt_size], dim=-1) # (B, K, 6)  ### Ingo: Is the heading direction inlcuded here? ###
        #position_embedding = PositionalEmbedding(geometric_feature, dim_g=self.dim_g) # (B,K,K, dim_g)
        position_embedding = self.pos_embedding(geometric_feature) # (B,K,K, dim_g)

        #position_embedding = self.feature_transform_pos(proposal_features)  #(B,appearance_feature_dim, K)
        #position_embedding = position_embedding.transpose(1, 2).contiguous() # (B, K, appearance_feature_dim)
        #transform proposal_features from 128-dim to appearance_feature_dim 
        #proposal_features = self.feature_transform1(proposal_features)  #(B,appearance_feature_dim, K)
        #proposal_features = proposal_features.transpose(1, 2).contiguous() # (B, K, appearance_feature_dim)
        #position_embedding = position_embedding.transpose(1, 2).contiguous() # (B, K, appearance_feature_dim)
        
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
            
            # Transformer Decoder Layer
            #query = self.decoder[i](query, key, query_pos, key_pos)

            # Prediction
            #base_xyz, base_size = self.prediction_heads[i](query,
            #                                               base_xyz=cluster_xyz,
            #                                               end_points=end_points,
            #                                               prefix=prefix)

            #base_xyz = base_xyz.detach().clone()
            #base_size = base_size.detach().clone()

        ### TODO: Add learning parameter to let the network decide during training if relation module is used ####
        #proposal_features = self.gamma * concat + f_a      # proposal_features: (B,K, appearance_feature_dim)
        proposal_features = concat
        #proposal_features = f_a + concat
        #proposal_features = torch.cat((f_a, concat), -1)
        
        #proposal_features = proposal_features.transpose(1,2).contiguous() #(B,appearance_feature_dim, K)
        proposal_features = self.feature_transform2(proposal_features) # (B,128,K)

        net = self.proposal_generation(proposal_features) # # (B, 2+3+num_heading_bin*2+num_size_cluster*4 + num_class, K)
        end_points = decode_scores(net, end_points, self.num_heading_bin, self.num_size_cluster)
        
        return end_points, proposal_features

    def init_weights(self):
        # initialize transformer
        #for m in self.relation.parameters():
        #    if m.dim() > 1:
        #        nn.init.xavier_uniform_(m)
        for m in self.feature_transform1.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.feature_transform2.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.proposal_generation.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        #for m in self.prediction_heads.parameters():
        #   if m.dim() > 1:
        #        nn.init.xavier_uniform_(m)


    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the width and height dimensions 
    x_shape = x.shape
    return torch.reshape(x, [x_shape[0], -1, x_shape[-1]]) # return [BATCH, W*H, CHANNELS]

class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=768,key_feature_dim = 96, geo_feature_dim = 96):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        #print(geo_feature_dim)
        #self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        #self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        #self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        #self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)

        '''Parameters'''
        #dim = 256
        #layer = dim//4

        '''Modules'''
        #self.gamma = nn.Parameter(torch.ones(1)) # requires_grad is True by default for Parameter
        #nn.init.constant_(self.gamma, 0.0)

        self.K = torch.nn.Conv2d(appearance_feature_dim, key_feature_dim, kernel_size=1, padding=0, stride=1, bias=False)
        self.Q = torch.nn.Conv2d(appearance_feature_dim, key_feature_dim, kernel_size=1, padding=0, stride=1, bias=False)
        self.V = torch.nn.Conv2d(appearance_feature_dim, appearance_feature_dim, kernel_size=1, padding=0, stride=1, bias=False)

        
        self.norm = nn.LayerNorm(appearance_feature_dim)
        self.dropout = nn.Dropout(0.1)


    def forward(self, f_a, position_embedding):#f_a: (B,K,appearance_feature_dim), position_embedding: (B,K,K,dim_g)
        
        f_a = f_a + position_embedding

        f_a = f_a.unsqueeze(-1)

        k = self.K(f_a)
        q = self.Q(f_a)
        v = self.V(f_a)

        k = k.transpose(dim0=1,dim1=3)
        q = q.transpose(dim0=1,dim1=3)
        v = v.transpose(dim0=1,dim1=3)

        s = torch.matmul(hw_flatten(q), hw_flatten(k).transpose(dim0=1,dim1=2))  # # [bs, N, N]

        beta = F.softmax(s, dim=-1)  # attention map

        o = torch.matmul(beta, hw_flatten(v))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        
        f_a = torch.squeeze(f_a, dim=-1)
        o = torch.reshape(o, shape=f_a.shape)  # [bs, h, w, C]
        
        #x = self.gamma * o + f_a
        x = self.dropout(o) + f_a
        x = x.permute(2, 0, 1)
        x = self.norm(x)
        x = x.permute(1, 2, 0)
        
        return x

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


#def init_weights(m):
#    if type(m) == nn.Linear or type(m) == nn.Conv1d:
#        gain = nn.init.calculate_gain('relu')
#        nn.init.xavier_uniform_(m.weight, gain=gain)
#        m.bias.data.fill_(0.01)
    #gain = nn.init.calculate_gain('relu')
    #nn.init.xavier_uniform_(m.weight, gain=gain)
    #m.bias.data.fill_(0.01)