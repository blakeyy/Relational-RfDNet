## Relational-RfD-Net 

**Relational-RfD-Net: A Semantic Instance Reconstruction framework using Attention** <br>
[Ingo Blakowski], [Trung Quoc Nguyen]<br>

### Setting up the project and basic commands
To set up the project please refer to the README_original.md of the original RfDNet in this project folder. There you can also see the basic commands to train and test the models.

---
### Training and testing
To control the training and testing the configuration files (see 'configs/config_files/****.yaml') are used.

#### Use the self-attention module:
1. Set before proposal generation MLP (before_prop_gen: True) 
2. Or/and set after proposal generation MLP (after_prop_gen: False):
   
   ```
   self_attention:
    appearance_feature_dim: 128
    before_prop_gen: True
    after_prop_gen: False
   ```

#### Use the relation-module (use_relation: True):
1. Use GT (use_gt_boxsize: True) or predicted (use_gt_boxsize: False) box size. 
2. Compute either two box losses before and after the relation module (compute_two_losses: True) or only one box loss after the relation module (compute_two_losses: False).
   ```
   relation_module:
    use_relation: False
    method: RelationalProposalModule
    loss: Null
    use_gt_boxsize: True
    compute_two_losses: False
    #use_learned_pos_embed: False
    n_relations: 8 #4
    appearance_feature_dim: 768 #384
    key_feature_dim: 96 
    geo_feature_dim: 96
    isDuplication: False
   ```
   