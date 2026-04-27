"""
20/4/26
Create bar graph summarizing results of APViT models.
Assumes results json file created by get_eval_results.py, with structure: {mdl_name: {repX: {test_X: Value}}}. with all
    test-blurs in test_blurs list below.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

file_name = 'results2'
model_names = ['RAF_blur0', 'RAF_blur8', 'RAF_blur0-8']  # choose from: ['RAF_blur0', 'RAF_blur4', 'RAF_blur0-4', 'RAF_blur8', 'RAF_blur0-8']
test_blurs = [0, 8]

with open(f"{file_name}.json", "r") as f:
    results = json.load(f)

f, axs = plt.subplots(1, len(test_blurs))
f.set_size_inches([4*len(test_blurs), 3.2])

for i, tb in enumerate(test_blurs):
    means = {mdl_nm: np.mean([results[k][r][f'test_{tb}'] for k in results.keys() for r in results[k].keys() if (mdl_nm == k)]) for mdl_nm in model_names}
    errs = {mdl_nm: np.std([results[k][r][f'test_{tb}'] for k in results.keys() for r in results[k].keys() if (mdl_nm == k)]) for mdl_nm in model_names}
    axs[i].bar(model_names, [means[m] for m in model_names], yerr=[errs[m] for m in model_names], align='center', alpha=0.5, ecolor='black')
    axs[i].set_ylim([0, 100])
    axs[i].set_title(f"Test Blur: {tb}")

plt.show(block=True)

print('done')
