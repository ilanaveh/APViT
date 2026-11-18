"""
20/4/26
Create bar graph summarizing results of APViT models.
Assumes results json file created by get_eval_results.py, with structure: {mdl_name: {repX: {test_X: Value}}}. with all
    test-blurs in test_blurs list below.
"""

import numpy as np
import matplotlib.pyplot as plt
import json


n_test_samples = 3068
yerr_capsize = 2

file_name = 'results'

plot_grouped = True
# 1. Models whose downstream training was the same as pretraining -> choose set_test_blur_by_RAF_blur = False below.
# model_names = ['RAF_blur0', 'RAF_blur8', 'RAF_blur0-8']  # choose from: ['RAF_blur0', 'RAF_blur4', 'RAF_blur0-4', 'RAF_blur8', 'RAF_blur0-8']

# 2. Models whose downstream training was either 0 / 8 regardless of pretraining -> choose set_test_blur_by_RAF_blur = True below.
freeze = True
# model_names = ['RAF_blur0', 'RAF_blur0_pretrained8', 'RAF_blur0_pretrained0-8',  # test blur: 0
#                'RAF_blur8_pretrained_blur0', 'RAF_blur8', 'RAF_blur8_pretrained0-8', 'RAF_blur8_pretrained8-16'  # test blur: 8
#                ]
model_names = {
    'Mixed': {
        'Baseline': 'RAF_blur8_pretrained0-8',
        'Forehead': 'RAF_blur8_mask_forehead_pretrained0-8',
        'Eyes': 'RAF_blur8_mask_eyes_pretrained0-8_freeze',
        'Nose': 'RAF_blur8_mask_nose_pretrained0-8',
        'Nose Narrow': 'RAF_blur8_mask_nose_narrow_pretrained0-8',
        'Mouth': 'RAF_blur8_mask_mouth_pretrained0-8',
        'Mouth+Eyes': 'RAF_blur8_mask_mouth_eyes_pretrained0-8',
        'Random': 'RAF_blur8_mask_random_patch_pretrained0-8'

    },
    'Low': {
        'Baseline': 'RAF_blur8',
        'Forehead': 'RAF_blur8_mask_forehead',
        'Eyes': 'RAF_blur8_mask_eyes',
        'Nose': 'RAF_blur8_mask_nose',
        'Nose Narrow': 'RAF_blur8_mask_nose_narrow',
        'Mouth': 'RAF_blur8_mask_mouth',
        'Mouth+Eyes': 'RAF_blur8_mask_mouth_eyes',
        'Random': 'RAF_blur8_mask_random_patch'
    }
}

width = 0.35
colors = {'Low': 'r', 'Mixed': 'g'}
pos = {'Low': - width / 2, 'Mixed': width/2}
labels = ('Baseline', 'Forehead', 'Eyes', 'Nose', 'Nose Narrow', 'Mouth', 'Mouth+Eyes', 'Random')

if freeze:
    for mdl_type, mdl_names in model_names.items():
        model_names[mdl_type] = {lbl: f'{m}_freeze' if 'freeze' not in m else m for (lbl, m) in mdl_names.items()}
        if 'RAF_blur8_pretrained0-8_freeze' in model_names[mdl_type]['Baseline']:
            model_names[mdl_type]['Baseline'] += '_fix_projs'
# test_blurs = [0, 8]
test_blurs = [8]
set_test_blur_by_RAF_blur = True

with open(f"{file_name}.json", "r") as f:
    results = json.load(f)

f, axs = plt.subplots(1, len(test_blurs))
f.set_size_inches([2 * len(model_names['Low']), 3.2])

if plot_grouped:
    ax = axs

    means = {mdl_type: [results[model_names[mdl_type][lbl]]['rep0'][f'test_{test_blurs[0]}'] for lbl in labels] for mdl_type in model_names.keys()}
    ps = {mdl_type: [results[model_names[mdl_type][lbl]]['rep0'][f'test_{test_blurs[0]}'] / 100 for lbl in labels]  for mdl_type in model_names.keys()}
    errs = {mdl_type: [100 * np.sqrt((p * (1-p)) / n_test_samples) for p in ps[mdl_type]] for mdl_type in model_names.keys()}

    x = np.arange(len(labels))

    rects = {}
    for mdl_type in model_names.keys():
        rects[mdl_type] = ax.bar(x + pos[mdl_type], means[mdl_type], width, label=mdl_type, yerr=errs[mdl_type],
                       align='center', alpha=0.5, ecolor='black', capsize=yerr_capsize, color=colors[mdl_type])

    ax.grid(axis='y')
    ax.set_ylim([40, 70])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
else:
    if set_test_blur_by_RAF_blur:
        means = {f'test_{tb}': {} for tb in test_blurs}
        errs = {f'test_{tb}': {} for tb in test_blurs}
        # result_lists = {f'test_{tb}': {mdl: [] for mdl in model_names if (f'RAF_blur{tb}' in mdl)} for tb in test_blurs}
        for tb in test_blurs:
            for mdl_nm in model_names:
                for k in results.keys():
                    if (mdl_nm == k) and (f'RAF_blur{tb}' in k):
                        means[f'test_{tb}'][mdl_nm] = results[k]['rep0'][f'test_{tb}']
                        # Compute err [sqrt( (p * (1-p)) / n)]:
                        p = results[k]['rep0'][f'test_{tb}'] / 100
                        errs[f'test_{tb}'][mdl_nm] = 100 * np.sqrt((p * (1-p)) / n_test_samples)
                    # for r in results[k].keys():
                    #     if (mdl_nm == k) and (f'RAF_blur{tb}' in k):
                    #         result_lists[f'test_{tb}'][mdl_nm].append(results[k][r][f'test_{tb}'])

        # means = {f'test_{tb}': {mdl: np.mean(result_lists[f'test_{tb}'][mdl]) for mdl in result_lists[f'test_{tb}'].keys()} for tb in test_blurs}
        # errs = {f'test_{tb}': {mdl: np.std(result_lists[f'test_{tb}'][mdl]) for mdl in result_lists[f'test_{tb}'].keys()} for tb in test_blurs}

        for i, tb in enumerate(test_blurs):
            ax = axs if len(test_blurs)==1 else axs[i]

            ax.grid(axis='y')

            bar_labels = [f"Train {k.split('blur')[-1].split('pretrained')[-1]}" for k in means[f'test_{tb}'].keys()]
            bar_labels =[k.split('mask_')[-1].split('_freeze')[0].split('_pretrained')[0] for k in means[f'test_{tb}'].keys()]
            ax.bar(bar_labels,
                       [means[f'test_{tb}'][m] for m in means[f'test_{tb}'].keys()],
                       yerr=[errs[f'test_{tb}'][m] for m in means[f'test_{tb}'].keys()],
                       align='center', alpha=0.5, ecolor='black', capsize=yerr_capsize, width=0.4, color='r')
            ax.set_ylim([40, 70])
            ax.set_title(f"Test Blur: {tb}")

    else:
        for i, tb in enumerate(test_blurs):
            # means = {mdl_nm: np.mean([results[k][r][f'test_{tb}'] for k in results.keys() for r in results[k].keys() if (mdl_nm == k)]) for mdl_nm in model_names}
            # errs = {mdl_nm: np.std([results[k][r][f'test_{tb}'] for k in results.keys() for r in results[k].keys() if (mdl_nm == k)]) for mdl_nm in model_names}
            # 28/4/26: Change means to acc of 1st rep, and errs to standard error over samples:
            means = {mdl_nm: results[mdl_nm]['rep0'][f'test_{tb}'] for mdl_nm in model_names}
            # Compute err [sqrt( (p * (1-p)) / n)]:
            ps = {mdl_nm: (results[mdl_nm]['rep0'][f'test_{tb}'] / 100) for mdl_nm in model_names}
            errs = {mdl_nm: 100 * np.sqrt((ps[mdl_nm] * (1-ps[mdl_nm])) / n_test_samples) for mdl_nm in model_names}
            axs[i].bar(model_names, [means[m] for m in model_names], yerr=[errs[m] for m in model_names], align='center', alpha=0.5, ecolor='black', capsize=yerr_capsize)
            axs[i].set_ylim([0, 100])
            axs[i].set_title(f"Test Blur: {tb}")

plt.show(block=True)

print('done')
