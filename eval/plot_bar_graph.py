"""
20/4/26
Create bar graph summarizing results of APViT models.
Assumes results json file created by deit_get_eval_results.py, with structure: {mdl_name: {repX: {test_X: Value}}}. with all
    test-blurs in test_blurs list below.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

file_name = 'results'
# model_names = ['RAF_blur0', 'RAF_blur8', 'RAF_blur0-8']  # choose from: ['RAF_blur0', 'RAF_blur4', 'RAF_blur0-4', 'RAF_blur8', 'RAF_blur0-8']
model_names = ['RAF_blur0', 'RAF_blur0_pretrained8', 'RAF_blur0_pretrained0-8',  # test blur: 0
               'RAF_blur8_pretrained_blur0', 'RAF_blur8', 'RAF_blur8_pretrained0-8'  # test blur: 8
               ]
test_blurs = [0, 8]
set_test_blur_by_RAF_blur = True

with open(f"{file_name}.json", "r") as f:
    results = json.load(f)

f, axs = plt.subplots(1, len(test_blurs))
f.set_size_inches([4*len(test_blurs), 3.2])

if set_test_blur_by_RAF_blur:
    means = {f'test_{tb}': {} for tb in test_blurs}
    result_lists = {f'test_{tb}': {mdl: [] for mdl in model_names if (f'RAF_blur{tb}' in mdl)} for tb in test_blurs}
    for tb in test_blurs:
        for mdl_nm in model_names:
            for k in results.keys():
                for r in results[k].keys():
                    if (mdl_nm == k) and (f'RAF_blur{tb}' in k):
                        result_lists[f'test_{tb}'][mdl_nm].append(results[k][r][f'test_{tb}'])

    means = {f'test_{tb}': {mdl: np.mean(result_lists[f'test_{tb}'][mdl]) for mdl in result_lists[f'test_{tb}'].keys()} for tb in test_blurs}
    errs = {f'test_{tb}': {mdl: np.std(result_lists[f'test_{tb}'][mdl]) for mdl in result_lists[f'test_{tb}'].keys()} for tb in test_blurs}

    for i, tb in enumerate(test_blurs):
        axs[i].bar([f"Train {k.split('blur')[-1].split('pretrained')[-1]}" for k in result_lists[f'test_{tb}'].keys()],
                   [means[f'test_{tb}'][m] for m in result_lists[f'test_{tb}'].keys()],
                   yerr=[errs[f'test_{tb}'][m] for m in result_lists[f'test_{tb}'].keys()],
                   align='center', alpha=0.5, ecolor='black')
        axs[i].set_ylim([60, 95])
        axs[i].set_title(f"Test Blur: {tb}")

else:
    for i, tb in enumerate(test_blurs):
        means = {mdl_nm: np.mean([results[k][r][f'test_{tb}'] for k in results.keys() for r in results[k].keys() if (mdl_nm == k)]) for mdl_nm in model_names}
        errs = {mdl_nm: np.std([results[k][r][f'test_{tb}'] for k in results.keys() for r in results[k].keys() if (mdl_nm == k)]) for mdl_nm in model_names}
        axs[i].bar(model_names, [means[m] for m in model_names], yerr=[errs[m] for m in model_names], align='center', alpha=0.5, ecolor='black')
        axs[i].set_ylim([0, 100])
        axs[i].set_title(f"Test Blur: {tb}")

plt.show(block=True)

print('done')
