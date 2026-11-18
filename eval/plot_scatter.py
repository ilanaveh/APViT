"""
15/6/26
Create scatter plot summarizing results of APViT models (difference from bar graph: results for multiple test blurs).
Assumes results json file created by get_eval_results.py, with structure: {mdl_name: {repX: {test_X: Value}}}. with all
    test-blurs in test_blurs list below.
"""

import numpy as np
import matplotlib.pyplot as plt
import json


n_test_samples = 3068
yerr_capsize = 2

file_name = 'results'

not_finiseh_yet = ['RAF_blur12_pretrained4', 'RAF_blur19_pretrained4', 'RAF_blur12_pretrained0-4', 'RAF_blur23_pretrained0-4',
                   # 'RAF_blur0_pretrained4_freeze',
                   'RAF_blur18_pretrained4_freeze', 'RAF_blur21_pretrained0-4_freeze',
                   # 'RAF_blur20_pretrained8', 'RAF_blur21_pretrained8',
                   ]

# test_blurs = np.arange(25)
test_blurs = np.arange(9)
# models = {f'RAF_blur{tb}_pretrained{pt}{fr_suf}': tb for tb in test_blurs for fr_suf in ['', '_freeze'] for pt in ['8', '0-8']}
models = {f'RAF_blur{tb}_pretrained{pt}{fr_suf}': tb for tb in test_blurs for fr_suf in ['_freeze'] for pt in ['8', '0-8', '0', '4', '0-4']}

for m in not_finiseh_yet:
    if m in models:
        models.pop(m)
if 'RAF_blur8_pretrained0-8_freeze' in models:
    models['RAF_blur8_pretrained0-8_freeze_fix_projs'] = models.pop('RAF_blur8_pretrained0-8_freeze')
if 'RAF_blur8_pretrained8_freeze' in models:
    models['RAF_blur8_freeze'] = models.pop('RAF_blur8_pretrained8_freeze')
if 'RAF_blur0_pretrained0_freeze' in models:
    models['RAF_blur0_freeze'] = models.pop('RAF_blur0_pretrained0_freeze')

means = {f'test_{tb}': {} for tb in test_blurs}
errs = {f'test_{tb}': {} for tb in test_blurs}

with open(f"{file_name}.json", "r") as f:
    results = json.load(f)

for mdl_nm, tb in models.items():
    for k in results.keys():
        if (mdl_nm == k) and (f'RAF_blur{tb}' in k):
            means[f'test_{tb}'][mdl_nm] = results[k]['rep0'][f'test_{tb}']
            # Compute err [sqrt( (p * (1-p)) / n)]:
            p = results[k]['rep0'][f'test_{tb}'] / 100
            errs[f'test_{tb}'][mdl_nm] = 100 * np.sqrt((p * (1-p)) / n_test_samples)

if 'RAF_blur8_pretrained0-8_freeze_fix_projs' in models:
    means['test_8']['RAF_blur8_pretrained0-8_freeze'] = means['test_8'].pop('RAF_blur8_pretrained0-8_freeze_fix_projs')
    means['test_8']['RAF_blur8_pretrained8_freeze'] = means['test_8'].pop('RAF_blur8_freeze')
    means['test_0']['RAF_blur0_pretrained0_freeze'] = means['test_0'].pop('RAF_blur0_freeze')
    errs['test_8']['RAF_blur8_pretrained0-8_freeze'] = errs['test_8'].pop('RAF_blur8_pretrained0-8_freeze_fix_projs')
    errs['test_8']['RAF_blur8_pretrained8_freeze'] = errs['test_8'].pop('RAF_blur8_freeze')
    errs['test_0']['RAF_blur0_pretrained0_freeze'] = errs['test_0'].pop('RAF_blur0_freeze')

test_keys = sorted(means.keys(), key=lambda k: int(k.split("_")[1]))

# Collect grouped results
grouped_means = {}
grouped_errors = {}

for test_key in test_keys:
    blur = int(test_key.split("_")[1])

    for model_name, value in means[test_key].items():
        group = model_name.split("pretrained", 1)[1]

        if group not in grouped_means:
            grouped_means[group] = {}
            grouped_errors[group] = {}

        grouped_means[group][blur] = value
        grouped_errors[group][blur] = errs[test_key][model_name]

plt.figure(figsize=(8, 5))

for group in sorted(grouped_means.keys()):
    x = sorted(grouped_means[group].keys())
    y = [grouped_means[group][blur] for blur in x]
    yerr = [grouped_errors[group][blur] for blur in x]

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=4,
        label=f"pretrained{group}",
    )

plt.xlabel("Test blur")
plt.ylabel("Accuracy")
plt.title("Model performance across test blur levels")
plt.xticks(test_blurs)
handles, labels = plt.gca().get_legend_handles_labels()
# order = [0, 2, 1, 3]
# reordered_handles = [handles[i] for i in order]
# reordered_labels = [labels[i] for i in order]
#
# # 4. Pass the reordered lists to plt.legend()
# plt.legend(reordered_handles, reordered_labels)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.ylim([40, 100])
plt.show(block=True)
print('done')
