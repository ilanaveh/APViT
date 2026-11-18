"""
7/7/26
Mask parts of the input image, and eval APViT models (that were trained without masks).
Based on get_eval_results.py.
"""

"""
26/4/26
Split plot_bar_graph.py - moved here code for running test and saving results.
"""
import mmcv
import json
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import test

save_breakdown = True

masks = {
    'eyes': 'quart2_offset10',
    'mouth': 'quart4_offset-15',
    'none': '',
}

save2file = True
append2existing = True
file_name = 'results_with_mask_breakdown' if save_breakdown else 'results_with_mask'

# blurs = np.arange(25)
# model_names = [f'RAF_blur{b}_pretrained{pt}{fr_suf}' for b in blurs for pt in ['8', '0-8'] for fr_suf in ['', '_freeze']]
# model_names = [m for m in model_names if ('blur8_pretrained8' not in m) and (m != 'RAF_blur8_pretrained0-8_freeze')]
model_names = ['RAF_blur8_freeze', 'RAF_blur8_pretrained0-8_freeze_fix_projs']
test_blurs = {m: None for m in model_names}  # None for using original config.
for m in model_names:
    test_blurs[m] = int(m.split('blur')[1].split('_')[0])

suf = '_fix_projs'  # uncomment for 'RAF_blur8_pretrained0-8_freeze_fix_projs' model.
# suf = ''  # comment for 'RAF_blur8_pretrained0-8_freeze_fix_projs' model.

if append2existing and os.path.isfile(f"{file_name}.json"):
    with open(f"{file_name}.json", "r") as f:
        results = json.load(f)
    for m in model_names:
        if m not in results.keys():
            results[m] = {msk: {} for msk in masks}
else:
    results = {m: {msk: {} for msk in masks} for m in model_names}


for m in model_names:

    print(m)
    cfg_pth = f'../configs/apvit/{m.replace(suf, "")}.py'
    cfg = mmcv.Config.fromfile(cfg_pth)

    for msk, msk_prt in masks.items():
        if f'test_{test_blurs[m]}' in results[m][msk]:
            continue

        args = [cfg_pth, f'../work_dirs/{m}/epoch_40.pth']

        if test_blurs[m] is not None:
            test_blur = test_blurs[m]
            args.extend(['--test_blur', str(test_blur)])
        else:
            test_blur = cfg.data['test']['pipeline'][1]['sigma_min'] if \
                cfg.data['test']['pipeline'][1]['type'] == 'GaussianBlur' else 0

        print(f"Testing with blur: {test_blur}")

        if save_breakdown:
            args.append('--return_labels')
        else:
            args.append('--return_acc')

        if msk_prt:
            args.extend(['--data_pth_suff', f'_masked_{msk_prt}'])

        if os.path.isfile(f'../work_dirs/{m}/epoch_40.pth'):
            cur_results = test.main(args)
            if save_breakdown:
                CLASSES = np.unique(cur_results['target_class'])
                cls2ind = {}
                for i, c in enumerate(cur_results['target_class']):
                    if c not in cls2ind:
                        cls2ind[c] = cur_results['target_label'][i]
                class_results = {cur_cls: [p == t for p, t in zip(cur_results['pred_label'], cur_results['target_label'])
                                           if t == cls2ind[cur_cls]] for cur_cls in CLASSES}
                results[m][msk][f'test_{test_blur}'] = {cur_cls: float(np.round(
                    sum(class_results[cur_cls]) / len(class_results[cur_cls]), 2))
                    for cur_cls in CLASSES}
                results[m][msk][f'test_{test_blur}']['total_acc'] = float(np.round(cur_results['top-1 accuracy'], 2))
            else:
                results[m][msk][f'test_{test_blur}'] = float(np.round(cur_results['top-1'], 2))

            if save2file:
                with open(f"{file_name}.json", "w") as f:
                    json.dump(results, f, indent=1)
        else:
            print(f"Training of model {m} hasn't finished. Moving to next model...")
