import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from net_utils.relation_tool import PositionalEmbedding, RankEmbedding
import net_utils.array_tool as at
from net_utils.bbox_tools import loc2bbox
from models.registers import MODULES

@MODULES.register_module
class DuplicationRemovalNetwork(nn.Module):
    def __init__(self,cfg, optim_spec = None):

        super(DuplicationRemovalNetwork, self).__init__()
        self.optim_spec = optim_spec
        self.cfg = cfg
        self.num_class = cfg.dataset_config.num_class
        self.appearance_feature_dim=128
        self.d_f = 128

        self.nms_rank_fc = nn.Linear(self.appearance_feature_dim, self.d_f, bias=True)
        self.feat_embedding_fc = nn.Linear(self.appearance_feature_dim,self.d_f,bias=True)

        self.attention = MODULES.get('SelfAttention')(cfg, optim_spec)

        self.nms_logit_fc = nn.Linear(self.d_f,1,bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,proposal_features, end_points):
        #proposal_features: (B,128,256) (B,128 ,K)
        #end_points: 
        B = proposal_features.shape[0]
        K = proposal_features.shape[2]
        proposal_features = proposal_features.transpose(2,1).contiguous()
        # cls_bbox: (B, K, NS, 6). NS = NC = number of classes
        mean_size_arr = self.cfg.mean_size_arr # (NS,3)
        pred_center = end_points['center'] # (B,K,3)
        size_residuals_normalized  = end_points['size_residuals_normalized'] # # (B,K,NS,3)
        mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3)
        pred_size = size_residuals_normalized*mean_size_arr_expanded + mean_size_arr_expanded # (B,K,NS,3)
        pred_center_expanded = pred_center.unsqueeze(2).repeat(1,1,self.num_class,1) # (B,K,NS,3)
        cls_bbox = torch.cat((pred_center_expanded,pred_size), dim=-1) # (B,K,NS,6)
        cls_bbox = cls_bbox.view(-1,self.num_class, 6) # (B*K,NS,6)

        # prob:  [B,K, NS]
        sem_cls_scores = end_points['sem_cls_scores'].view(-1,self.num_class) # (B*K,NS)
        prob = F.softmax(sem_cls_scores, dim=-1) # [B*K,NS]
        


        prob,prob_argmax = torch.max(prob,dim=-1) # prob: [B*K], prob_argmax: [B*K,] value in 0...NS-1
        cls_bbox = cls_bbox[np.arange(start=0,stop=prob.shape[0]),prob_argmax] # [B*K, 6] with each bounding box, take the bounding box of the class that has highest probability
        
        prob = prob.view(B,K) # (B,K)
        prob_argmax = prob_argmax.view(B,K)
        cls_bbox = cls_bbox.view(B,K,6)

        sorted_score,prob_argsort = torch.sort(prob,descending=True) # prob_argsort: (B,K), value from 0...K-1

        sorted_prob = sorted_score# [B,K]
        sorted_cls_bboxes = torch.gather(cls_bbox, 1,prob_argsort.unsqueeze(-1).repeat(1,1,6)) # [B,K,6]
        sorted_labels =  torch.gather(prob_argmax,1,prob_argsort) # [B,K] 
        sorted_features = torch.gather(proposal_features,1,prob_argsort.unsqueeze(-1).repeat(1,1,self.appearance_feature_dim)) # [B,K, appearance_feature_dim]

        nms_rank_embedding = RankEmbedding(sorted_prob.size()[1],self.appearance_feature_dim) # [K,appearance_feature_dim]
        nms_rank = self.nms_rank_fc(nms_rank_embedding) # [K,d_f]
        nms_rank = nms_rank.unsqueez(0).repeat(B,1,1) # (B,K,d_f)

        roi_feat_embedding = self.feat_embedding_fc(sorted_features) # [B,K,d_f]
        nms_embedding_feat = nms_rank + roi_feat_embedding # [B,K,d_f]
        nms_embedding_feat = nms_embedding_feat.transpose(2,1).contiguous() # [B,d_f,K]

        nms_logit = self.attention(nms_embedding_feat) # [B,d_f,K] (B,128,K)
        nms_logit = nms_logit.transpose(2,1).contiguous() # (B,K,128)
        nms_logit = self.nms_logit_fc(nms_logit) # [B,K,1]
        s1 = self.sigmoid(nms_logit).view(B,K)
        nms_scores = s1 * sorted_prob # [B,K]
        end_points['duplicate_removal_scores'] = nms_scores
        # nms_scores: [B,K], s1*s0 for each predicted object
        # sorted_labels: [B,K], class labels, value in 0...num_classes-1, of the corresponding predicted objects. 
        # sorted_cls_bboxes: [B,K,6]: bounding box parameters of each predicted object. 
        return nms_scores, sorted_labels, sorted_cls_bboxes
