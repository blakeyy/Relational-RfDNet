from net_utils.registry import Registry
from models.registers import MODULES
import torch
import torch.nn as nn
from torch.nn import functional as F
from net_utils.nn_distance import nn_distance
import numpy as np
from net_utils.box_util import get_3d_box
from net_utils.relation_tool import PositionalEmbedding
from net_utils.nn_distance import nn_distance

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch._C._TensorBase):
        return data.cpu().numpy()
    if isinstance(data, torch.autograd.Variable):
        return tonumpy(data.data)

def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch._C._TensorBase):
        tensor = data
    if isinstance(data, torch.autograd.Variable):
        tensor = data.data
    if cuda:
        tensor = tensor.cuda()
    return tensor

def tovariable(data):
    if isinstance(data, np.ndarray):
        return tovariable(totensor(data))
    if isinstance(data, torch._C._TensorBase):
        return torch.autograd.Variable(data)
    if isinstance(data, torch.autograd.Variable):
        return data
    else:
        raise ValueError("UnKnow data type: %s, input should be {np.ndarray,Tensor,Variable}" %type(data))

def RankEmbedding(rank_dim=128,feat_dim=1024,wave_len=1000):
    rank_range = torch.arange(0, rank_dim).cuda().float()

    feat_range = torch.arange(feat_dim / 2).cuda()
    dim_mat = feat_range / (feat_dim / 2)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, -1)
    rank_mat = rank_range.view(-1, 1)

    mul_mat = rank_mat * dim_mat
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

    return embedding

def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.
    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.
    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.
    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`
    The decoding formulas are used in works such as R-CNN [#]_.
    The output is same type as the type of the inputs.
    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.
    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.
    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.
    """

    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox

@MODULES.register_module
class DuplicationRemovalNetwork(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(DuplicationRemovalNetwork, self).__init__()
    
        self.optim_spec = optim_spec
        self.cfg = cfg
        n_relations = self.cfg.config['model']['enhance_recognition']['n_relations']
        
        self.appearance_feature_dim = self.cfg.config['model']['detection']['appearance_feature_dim']
        
        d_f = 128

        self.n_class = cfg.dataset_config.num_class

        self.nms_rank_fc = nn.Linear(self.appearance_feature_dim, d_f, bias=True)
        self.roi_feat_embedding_fc = nn.Linear(self.appearance_feature_dim, d_f, bias=True)

        self.attention = MODULES.get('SelfAttention')(cfg) ### TODO

        self.nms_logit_fc = nn.Linear(self.appearance_feature_dim,1,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, end_points, appearance_features):  ### TODO

        config_dict = self.cfg.eval_config

        pred_size_class = torch.argmax(end_points['size_scores'], -1)  # B,num_proposal
        size_residuals = end_points['size_residuals_normalized'] * torch.from_numpy(config_dict['dataset_config'].mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        pred_size_residual = torch.gather(size_residuals, 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3))  # B,num_proposal,1,3
        pred_size_residual.squeeze_(2)
        mean_size_arr = torch.from_numpy(config_dict['dataset_config'].mean_size_arr.astype(np.float32)).cuda()
        pred_size_class = torch.squeeze(pred_size_class.type(torch.cuda.LongTensor)) ## Problem if batch_size==1 -> change where to squeeze
                
        box_size = mean_size_arr[pred_size_class, :] + pred_size_residual
        center = end_points['center'] # (B, K, 3)               # (B, num_proposal, 3)
        sem_cls_scores = end_points['sem_cls_scores']            # (B, num_proposal, 8)

        N = sem_cls_scores.shape[-1]
        
        prob = F.softmax(tovariable(sem_cls_scores), dim=1)

        prob, prob_argmax = torch.max(prob,dim=-1)
        cls_bbox = box_size[np.arange(start=0, stop=N), prob_argmax]

        nonzero_idx = torch.nonzero(prob_argmax)

        if(nonzero_idx.size()[0]==0):
            return None, None, None
        else:
            nonzero_idx = nonzero_idx[:, 0]
            prob_argmax = prob_argmax[nonzero_idx]
            prob = prob[nonzero_idx]
            cls_bbox = cls_bbox[nonzero_idx]
            appearance_features_nobg = appearance_features[nonzero_idx]
            sorted_score, prob_argsort = torch.sort(prob, descending=True)

            sorted_prob = prob[prob_argsort]
            sorted_cls_bboxes = cls_bbox[prob_argsort]
            sorted_labels =  prob_argmax[prob_argsort]
            sorted_features = appearance_features_nobg[prob_argsort]

            nms_rank_embedding = RankEmbedding(sorted_prob.size()[0], self.appearance_feature_dim)   ### TODO
            nms_rank = self.nms_rank_fc(nms_rank_embedding)
            roi_feat_embedding = self.roi_feat_embedding_fc(sorted_features)
            nms_embedding_feat = nms_rank + roi_feat_embedding
            #position_embedding = PositionalEmbedding(sorted_cls_bboxes,dim_g = self.geo_feature_dim)
            #nms_logit = self.relation_module([sorted_features, nms_embedding_feat,position_embedding])
            nms_logit = self.attention([sorted_features, nms_embedding_feat])       ### TODO
            nms_logit = self.nms_logit_fc(nms_logit)
            s1 = self.sigmoid(nms_logit).view(-1)
            nms_scores = s1 * sorted_prob

            return nms_scores, sorted_labels-1, sorted_cls_bboxes