import torch

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
def PositionalEmbedding_old( f_g, dim_g=64, wave_len=1000):
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(1, -1))
    delta_h = torch.log(h / h.view(1, -1))
    size = delta_h.size()

    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_w = delta_w.view(size[0], size[1], 1)
    delta_h = delta_h.view(size[0], size[1], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

    return embedding
def PositionalEmbedding( f_g, dim_g=96, wave_len=1000):
    """
    Args:
        f_g: (B, K, 6)
    Returns:
        embedding: (B, K, K, dim_g)
    """
    batch_size = f_g.shape[0]
    cx, cy, cz, l,w,h = torch.chunk(f_g, 6, dim=2) # (B,K,1)

    delta_x = cx - cx.view(batch_size, 1, -1) #  (B,K,K)
    delta_x = torch.clamp(torch.abs(delta_x / l), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / w), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_z = cz - cz.view(batch_size, 1, -1)
    delta_z = torch.clamp(torch.abs(delta_z / h), min=1e-3)
    delta_z = torch.log(delta_z)

    delta_l = torch.log(l.view(batch_size, 1,-1) / l)
    delta_w = torch.log(w.view(batch_size, 1,-1) / w)
    delta_h = torch.log(h.view(batch_size, 1,-1) / h)
    size = delta_h.size() # (B,K,K)

    delta_x = delta_x.view(size[0], size[1], size[2], 1)
    delta_y = delta_y.view(size[0], size[1], size[2], 1)
    delta_z = delta_z.view(size[0], size[1], size[2], 1)
    delta_l = delta_l.view(size[0], size[1], size[2], 1)
    delta_w = delta_w.view(size[0], size[1], size[2], 1)
    delta_h = delta_h.view(size[0], size[1], size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_z, delta_l, delta_w, delta_h), -1) # (B,K,K,6)

    feat_range = torch.arange(dim_g / 12).cuda()
    dim_mat = feat_range / (dim_g / 12)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))
    
    dim_mat = dim_mat.view(1,1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], size[2], 6, -1) # (B,K,K,6,1)
    position_mat = 100. * position_mat
    mul_mat = position_mat * dim_mat 

    mul_mat = mul_mat.view(size[0], size[1], size[2], -1) # (B,K,K,48)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1) # (B,K,K,dim_g). Note that 1st K is key and 2nd K is query

    return embedding