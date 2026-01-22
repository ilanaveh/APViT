"""
21/1/26
Start from inference code from apvit git (demo.ipynb)
While inference, attn_map is extracted from vit_siam_merge>PoolingVit>forward_features, but
it isn't returned later (by APViT/mmcls/models/classifiers/pool_vit.py>extract_feat).
=> I'm using APViT/mmcls/models/classifiers/pool_vit.py>extract_attn_map directly.
"""

import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmcls.models.utils import top_pool


from mmcls.models.builder import build_classifier
from mmcls.datasets.raf import FER_CLASSES
from mmcls.datasets.pipelines import Compose
from mmcls.apis.inference import init_model, inference_model

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

cfg = mmcv.Config.fromfile("configs/apvit/RAF.py")
cfg.model.pretrained = None
cfg.model.extractor.pretrained = None
cfg.model.vit.pretrained = None
cfg.model.vit.cnn_pool_config['keep_num'] = 10

# build the model and load checkpoint
classifier = build_classifier(cfg.model)
load_checkpoint(classifier, "weights/APViT_RAF-3eeecf7d.pth", map_location='cpu')
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

img = mmcv.imread('resources/demo.jpg')

plt.imshow(img[:, :, ::-1])

# preprocess the image
data = test_preprocess(dict(img=img))
data['img'] = data['img'][None, ...].cuda()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Visualize Attention: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
attn_map = classifier.extract_attn_map(data['img'])
attn_map_np = attn_map.detach().cpu().numpy().squeeze()
att_112 = np.kron(attn_map_np, np.ones((8, 8), dtype=int))
# 1. Show attention-map on image:
plt.figure(figsize=(6, 6))
plt.imshow(img[:, :, [2, 1, 0]])  # switch from BGR to RGB
plt.imshow(att_112, cmap='jet', alpha=.4)

# 2. Remove low-attention patches, and show result:
# Code based on APViT.mmcls.models.vit.vit_siam_merge.PoolingViT.forward_features.
plt.figure(figsize=(6, 6))
attn_weight = attn_map.flatten(2).transpose(2, 1)
if cfg.model.vit.cnn_pool_config is not None:
    keep_indexes = top_pool(attn_weight, dim=1, **cfg.model.vit.cnn_pool_config).squeeze()
    if keep_indexes is not None:
        mask = np.zeros(attn_weight.squeeze().shape)
        mask[keep_indexes.cpu()] = True
        assert(mask.sum() == cfg.model.vit.cnn_pool_config['keep_num'])
        mask_14 = mask.reshape(14, 14)
        mask_112 = np.kron(mask_14, np.ones((8, 8), dtype=int))
        img_masked = img.copy()
        img_masked[~mask_112.astype(bool)] = 255
        plt.imshow(img_masked[:, :, [2, 1, 0]])
        # plt.imshow(att_112, cmap='jet', alpha=.4)

plt.show(block=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ run the inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
out = classifier(**data, return_loss=False)
result_index = np.argmax(out[0])

print(f'Predict result: {FER_CLASSES[result_index]} with confidance: {out[0][result_index]:.2f}')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Alternative method for inference: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model = init_model(
    config='configs/apvit/RAF.py',
    checkpoint='weights/APViT_RAF-3eeecf7d.pth'
)

result = inference_model(model, img)
result
