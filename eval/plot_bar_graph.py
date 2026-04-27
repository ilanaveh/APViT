"""
20/4/26
Create bar graph summarizing results of APViT models.
Assumes results json file created by get_eval_results.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

file_name = 'results2'
model_names = ['RAF_blur0', 'RAF_blur0-8', 'RAF_blur8']  # choose from: ['RAF_blur0', 'RAF_blur4', 'RAF_blur0-4', 'RAF_blur8', 'RAF_blur0-8']
test_blur = 0

with open(f"{file_name}.json", "r") as f:
    results = json.load(f)

means = {mdl_nm: np.mean([results[k][r][f'test_{test_blur}'] for k in results.keys() for r in results[k].keys() if (mdl_nm == k)]) for mdl_nm in model_names}
errs = {mdl_nm: np.std([results[k][r][f'test_{test_blur}'] for k in results.keys() for r in results[k].keys() if (mdl_nm == k)]) for mdl_nm in model_names}
f, ax = plt.subplots()
ax.bar(model_names, [means[m] for m in model_names], yerr=[errs[m] for m in model_names], align='center', alpha=0.5, ecolor='black')
ax.set_ylim([0, 100])
plt.show(block=True)

print('done')
