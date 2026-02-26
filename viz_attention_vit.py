"""
26/1/26
Based on viz_attention_cnn.
"""
import numpy as np
import matplotlib.pyplot as plt
import mmcv
from mmcv.runner import load_checkpoint
from mmcls.models.builder import build_classifier
from mmcls.datasets.pipelines import Compose
from mmcls.models.utils import top_pool


cfg = mmcv.Config.fromfile("configs/apvit/RAF.py")
cfg.model.pretrained = None
cfg.model.extractor.pretrained = None
cfg.model.vit.pretrained = None
cfg.model.vit.cnn_pool_config['keep_num'] = 160
cfg.model.vit.vit_pool_configs['keep_rates'] = [1.] * 4 + [0.9] * 4
cfg.model.vit.attn_before_proj = True

# build the model and load checkpoint
classifier = build_classifier(cfg.model)
# load_checkpoint(classifier, "weights/APViT_RAF-3eeecf7d.pth", map_location='cpu')  # official pretrained
load_checkpoint(classifier, "work_dirs/RAF_blur8/epoch_40.pth", map_location='cpu')

classifier = classifier.to("cuda")
classifier.eval()

# define the preprocess for test
test_preprocess = Compose([
    dict(type='Resize', size=112),
    dict(type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img',])
])

# img = mmcv.imread('resources/test_0002_112.jpg')  # choose between 'demo.jpg' / 'test_0002_112.jpg'
img = mmcv.imread('resources/demo.jpg')

# preprocess the image
data = test_preprocess(dict(img=img))
data['img'] = data['img'][None, ...].cuda()

# Get vit attention-maps per block:
vit_attn_weights = []
vit_keep_inds = []


# To enable following hooks, I added 'self.keep_index' and 'self.attn_weight' properties to PoolingAttention
# (implemented in APViT.mmcls.models.backbones.modules.vit_pooling.PoolingAttention.forward)

def hook_attn_fn(module, input, output):
    vit_attn_weights.append(module.attn_weight.cpu())


def hook_keep_ind_fn(module, input, output):
    if module.keep_index is not None:
        vit_keep_inds.append(module.keep_index.cpu())
    else:
        vit_keep_inds.append(module.keep_index)


for block in classifier.vit.blocks:
    block.attn.register_forward_hook(hook_attn_fn)
    block.attn.register_forward_hook(hook_keep_ind_fn)


# First, extract CNN attention, for consistency with indexes (vit indexes are selected only from remains from cnn):
cnn_attn_map = classifier.extract_attn_map(data['img'])
cnn_attn_weight = cnn_attn_map.flatten(2).transpose(2, 1)
if cfg.model.vit.cnn_pool_config is not None:
    cnn_keep_indexes = top_pool(cnn_attn_weight, dim=1, **cfg.model.vit.cnn_pool_config).squeeze().cpu()

# Forward pass:
# out = classifier(**data, return_loss=False)

# Put attention_weight values in correct positions (in original 196 vector)
current_keep_inds = cnn_keep_indexes
for blk in range(8):
    plt.figure(figsize=(6, 6))

    full_attn_weight = np.zeros(196)
    if blk <= 4:  # First five blocks - no need to remove additional patches:
        full_attn_weight[current_keep_inds] = vit_attn_weights[blk].detach().squeeze()[1:]  # remove cls token [index 0]
    else:  # Last three blocks - need to remove additional patches:
        current_keep_inds = current_keep_inds[vit_keep_inds[blk - 1].detach().squeeze()[1:] - 1]  # -1 since 1st was cls token.
        full_attn_weight[current_keep_inds] = vit_attn_weights[blk].detach().squeeze()[1:]
    attn_map = full_attn_weight.reshape([14, 14])
    att_112 = np.kron(attn_map, np.ones((8, 8), dtype=int))  # duplicate each attention value to all pixels in patch

    mask = (att_112 == 0)   # for creating masked_attn - so color map takes into account only the relevant patches.
    masked_att = np.ma.masked_where(mask, att_112)

    img_masked = img.copy()
    img_masked[~att_112.astype(bool)] = 255  # mask patches that were already removed by CNN (in white)

    plt.imshow(img_masked[:, :, ::-1])
    plt.imshow(masked_att, cmap='jet', alpha=.4)

    plt.title(f'Block {blk}')

plt.show(block=True)
