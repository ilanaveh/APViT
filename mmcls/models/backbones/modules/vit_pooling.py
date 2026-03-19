import torch
from torch import nn
from .vit import Mlp
from mmcls.models.utils import top_pool
from mmcls.models.vit.layers import DropPath


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                 pool_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pool_config = pool_config


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, head_num, token_num, token_num]

        if self.pool_config:
            attn_method = self.pool_config.get('attn_method')
            if attn_method == 'SUM_ABS_1':
                attn_weight = attn[:, :, 0, :].transpose(-1, -2)    # [B, token_num, head_num]
                attn_weight = torch.sum(torch.abs(attn_weight), dim=-1).unsqueeze(-1)
            elif attn_method == 'SUM':
                attn_weight = attn[:, :, 0, :].transpose(-1, -2)    # [B, token_num, head_num]
                attn_weight = torch.sum(attn_weight, dim=-1).unsqueeze(-1)
            elif attn_method == 'MAX':
                attn_weight = attn[:, :, 0, :].transpose(-1, -2)
                attn_weight = torch.max(attn_weight, dim=-1)[0].unsqueeze(-1)
            else:
                raise ValueError('Invalid attn_method: %s' % attn_method)

            # attn_weight = torch.rand(attn_weight.shape, device=attn_weight.device)
            keep_index = top_pool(attn_weight, dim=self.dim, **self.pool_config)
        else:
            keep_index = None

        # IN 26/1/26: Add for visualizing attention (viz_attention_vit.py)
        self.attn_weight = attn_weight
        if keep_index is not None:
            self.keep_index = top_pool(attn_weight, dim=1, **self.pool_config)  # dim=1
        else:
            self.keep_index = None

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, keep_index


class PoolingBlock(nn.Module):

    def __init__(self, dim=0, num_heads=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_config=None, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            pool_config=pool_config)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        feature, keep_index = self.attn(self.norm1(x))
        x = x + self.drop_path(feature)
        if keep_index is not None:
            if len(keep_index) != x.shape[1]:
                x = x.gather(dim=1, index=keep_index)
                # pooled_x = []
                # for i in range(keep_index.shape[0]):
                #     pooled_x.append(x[i, keep_index[i, :, 0]])
                # x = torch.stack(pooled_x)
                # assert torch.all(torch.eq(quick_x, x))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class StudentPoolingBlock(PoolingBlock):
    """
    Based on PoolingBlock.
    Used for incorporating option for distillation.
    Changes:
        * init:
            * Add use_kd bool as optional arg.
            * self.attn = StudentPoolingAttention (instead of PoolingAttention).
        * forward: add tchr_attn_weights tchr_keep_index as optional args, and pass them to self.attn.
    """

    def __init__(self, dim=0, num_heads=0, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., pool_config=None,
                 use_kd=False, *args, **kwargs):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop,
                         pool_config=pool_config, *args, **kwargs)
        self.attn = StudentPoolingAttention(
            dim, use_kd=use_kd, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, pool_config=pool_config)

    def forward(self, x, tchr_attn_weight=None, tchr_keep_index=None):
        feature, keep_index = self.attn(self.norm1(x), tchr_attn_weight, tchr_keep_index)
        x = x + self.drop_path(feature)
        if keep_index is not None:
            if len(keep_index) != x.shape[1]:
                x = x.gather(dim=1, index=keep_index)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class StudentPoolingAttention(PoolingAttention):
    """
    Based on PoolingAttention.
    Use logic similar to 'StudentPoolingViT' (in vit_siam_merge) for incorporating option for distillation:
        1. Add use_kd to init.
        2. Modify forward to allow using tchr_attn_weight for pooling.
    """
    def __init__(self, dim, use_kd=False, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.use_kd = use_kd

    def forward(self, x, tchr_attn_weight_vit=None, tchr_keep_index=None):
        """
        Bosed on PoolingAttention forward.
        Changes (Same logic as 'StudentPoolingViT' (in vit_siam_merge) ):
            1. [Removed additions for attention-visualization).]
            2. Optional arg: tchr_attn_weight_vit - Transformer attention-weight from teacher model.
                * as opposed to cnn, here the input is the attention-weight (not map), just because I already have the
                    hook in vit_pooling.PoolingAttention.
            3. Use tchr_attn_weight_vit instead of attn_weight, if self.use_kd = True.
            4. For validation: optional arg 'tchr_keep_index' - compare to what is calculated here.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, head_num, token_num, token_num]

        if self.pool_config:
            if self.use_kd and self.training:
                attn_weight = tchr_attn_weight_vit
            else:
                attn_method = self.pool_config.get('attn_method')
                if attn_method == 'SUM_ABS_1':
                    attn_weight = attn[:, :, 0, :].transpose(-1, -2)    # [B, token_num, head_num]
                    attn_weight = torch.sum(torch.abs(attn_weight), dim=-1).unsqueeze(-1)
                elif attn_method == 'SUM':
                    attn_weight = attn[:, :, 0, :].transpose(-1, -2)    # [B, token_num, head_num]
                    attn_weight = torch.sum(attn_weight, dim=-1).unsqueeze(-1)
                elif attn_method == 'MAX':
                    attn_weight = attn[:, :, 0, :].transpose(-1, -2)
                    attn_weight = torch.max(attn_weight, dim=-1)[0].unsqueeze(-1)
                else:
                    raise ValueError('Invalid attn_method: %s' % attn_method)

                # attn_weight = torch.rand(attn_weight.shape, device=attn_weight.device)
            keep_index = top_pool(attn_weight, dim=self.dim, **self.pool_config)

            if self.use_kd and (keep_index is not None) and self.training:
                assert torch.all(keep_index == tchr_keep_index)
        else:
            keep_index = None

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, keep_index
