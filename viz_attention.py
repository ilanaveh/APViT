"""
21/1/26
Start from inference code from apvit git (demo.ipynb)
ToDo: add code for hooking attention-map and visualizing it.
"""

import numpy as np
import mmcv
from mmcv.runner import load_checkpoint

from mmcls.models.builder import build_classifier
from mmcls.datasets.raf import FER_CLASSES
from mmcls.datasets.pipelines import Compose
from mmcls.apis.inference import init_model, inference_model

import matplotlib.pyplot as plt


cfg = mmcv.Config.fromfile("configs/apvit/RAF.py")
cfg.model.pretrained = None
cfg.model.extractor.pretrained = None
cfg.model.vit.pretrained = None

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

# run the inference
out = classifier(**data, return_loss=False)
result_index = np.argmax(out[0])

print(f'Predict result: {FER_CLASSES[result_index]} with confidance: {out[0][result_index]:.2f}')

model = init_model(
    config='configs/apvit/RAF.py',
    checkpoint='weights/APViT_RAF-3eeecf7d.pth'
)

result = inference_model(model, img)
result
