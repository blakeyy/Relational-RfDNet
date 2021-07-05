# Proposal module in votenet.
# author: ynie
# date: March, 2020
# cite: VoteNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.registers import MODULES
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import  PointnetSAModuleVotes, PointnetFPModule
from external.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

def decode_scores(net, end_points, num_heading_bin, num_size_cluster):
    net_transposed = net.transpose(2, 1)  # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    # objectness
    objectness_scores = net_transposed[:, :, 0:2]
    end_points['objectness_scores'] = objectness_scores

    # center
    base_xyz = end_points['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:, :, 2:5]  # (batch_size, num_proposal, 3)
    end_points['center'] = center

    # heading
    heading_scores = net_transposed[:, :, 5:5 + num_heading_bin]
    heading_residuals_normalized = net_transposed[:, :, 5 + num_heading_bin:5 + num_heading_bin * 2]
    end_points['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized  # Bxnum_proposalxnum_heading_bin (should be -1 to 1)

    # size
    size_scores = net_transposed[:, :, 5 + num_heading_bin * 2:5 + num_heading_bin * 2 + num_size_cluster]
    size_residuals_normalized = net_transposed[:, :,
                                5 + num_heading_bin * 2 + num_size_cluster:5 + num_heading_bin * 2 + num_size_cluster * 4].view(
        [batch_size, num_proposal, num_size_cluster, 3])  # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized

    sem_cls_scores = net_transposed[:, :, 5 + num_heading_bin * 2 + num_size_cluster * 4:]  # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    return end_points


@MODULES.register_module
class ProposalModule(nn.Module):
    def __init__(self, cfg, optim_spec = None):
        '''
        Skeleton Extraction Net to obtain partial skeleton from a partial scan (refer to PointNet++).
        :param config: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(ProposalModule, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        '''Parameters'''
        self.num_class = cfg.dataset_config.num_class
        self.num_heading_bin = cfg.dataset_config.num_heading_bin
        self.num_size_cluster = cfg.dataset_config.num_size_cluster
        self.mean_size_arr = cfg.dataset_config.mean_size_arr
        self.num_proposal = cfg.config['data']['num_target']
        self.sampling = cfg.config['data']['cluster_sampling']
        self.seed_feat_dim = 256
        feat_dim = self.cfg.config['model']['detection']['appearance_feature_dim']

        '''Modules'''
        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                #mlp=[self.seed_feat_dim, 128, 128, 128],
                mlp=[self.seed_feat_dim, feat_dim, feat_dim, feat_dim],
                use_xyz=True,
                normalize_xyz=True
            )

        # MLPs
        #self.mlp1 = nn.Sequential(nn.Conv1d(128,feat_dim,1), \
        #                                    nn.BatchNorm1d(feat_dim), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(feat_dim,feat_dim,1))
        #self.mlp2 = nn.Sequential(nn.Conv1d(feat_dim,feat_dim,1), \
        #                                    nn.BatchNorm1d(feat_dim), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(feat_dim,feat_dim,1))  
        #self.mlp3 = nn.Sequential(nn.Conv1d(feat_dim,feat_dim,1), \
        #                                    nn.BatchNorm1d(feat_dim), \
        #                                    nn.ReLU(), \
        #                                    nn.Conv1d(feat_dim,128,1)) 
        
         

        # Attention Module
        if cfg.config['model']['detection']['use_attention']:
            self.attention1 = MODULES.get('SelfAttention')(cfg, optim_spec)
            self.attention2 = MODULES.get('SelfAttention')(cfg, optim_spec)
            self.attention3 = MODULES.get('SelfAttention')(cfg, optim_spec)
            self.attention4 = MODULES.get('SelfAttention')(cfg, optim_spec)

            self.attention5 = MODULES.get('SelfAttention')(cfg, optim_spec)
            self.attention6 = MODULES.get('SelfAttention')(cfg, optim_spec)
            self.attention7 = MODULES.get('SelfAttention')(cfg, optim_spec)
            self.attention8 = MODULES.get('SelfAttention')(cfg, optim_spec)

            #self.mlp4 = nn.Sequential(nn.Conv1d(512,256,1), \
            #                        nn.BatchNorm1d(256), \
            #                        nn.ReLU(), \
            #                        nn.Conv1d(256,128,1))

            #self.gn = torch.nn.GroupNorm(8, 1024)

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        #self.conv1 = torch.nn.Conv1d(feat_dim,128,1) 
        self.conv1 = torch.nn.Conv1d(1152,512,1)
        self.conv2 = torch.nn.Conv1d(512,256,1)
        self.conv3 = torch.nn.Conv1d(256,2+3+self.num_heading_bin*2+self.num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)


    def forward(self, xyz, features, end_points, export_proposal_feature=False):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            self.cfg.log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- SELF-ATTENTION MODULE ---------
        if self.cfg.config['model']['detection']['use_attention']:
            
            #features = self.mlp1(features)
            #features = self.attention1([features, None])
            #features = self.attention2([features, None])
            #features = self.attention3([features, None])
            #features = self.attention4([features, None])
            #features = self.mlp2(features)
            #features = self.attention5([features, None])
            #features = self.attention6([features, None])
            #features = self.attention7([features, None])
            #features = self.attention8([features, None])
            #features = self.mlp3(features)
            
            input = features
            features = torch.cat((input, self.attention1([input, None])), 1)
            features = torch.cat((features, self.attention2([input, None])), 1)
            features = torch.cat((features, self.attention3([input, None])), 1)
            features = torch.cat((features, self.attention4([input, None])), 1)

            features = torch.cat((features, self.attention5([input, None])), 1)
            features = torch.cat((features, self.attention6([input, None])), 1)
            features = torch.cat((features, self.attention7([input, None])), 1)
            features = torch.cat((features, self.attention8([input, None])), 1)
            #features = self.gn(features)
            
            #features = self.mlp4(features)
            #features = self.attention1([features, None])

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        end_points = decode_scores(net, end_points, self.num_heading_bin, self.num_size_cluster)

        if export_proposal_feature:
            return end_points, features
        else:
            return end_points, None