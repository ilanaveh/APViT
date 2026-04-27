"""
20/4/26
Create bar graph summarizing results of APViT models.
Assumes results json file created by get_eval_results.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

file_name = 'results2'
model_names = ['RAF_blur0', 'RAF_blur8', 'RAF_blur0-8']  # choose from: ['RAF_blur0', 'RAF_blur4', 'RAF_blur0-4', 'RAF_blur8', 'RAF_blur0-8']

with open(f"{file_name}.json", "r") as f:
    results = json.load(f)

means = {mdl_nm: np.mean([results[k]['top-1'] for k in results.keys() if ((mdl_nm == k) or (f'{mdl_nm}_rep' in k))]) for mdl_nm in model_names}
errs = {mdl_nm: np.std([results[k]['top-1'] for k in results.keys() if ((mdl_nm == k) or (f'{mdl_nm}_rep' in k))]) for mdl_nm in model_names}
f, ax = plt.subplots()
ax.bar(model_names, [means[m] for m in model_names], yerr=[errs[m] for m in model_names], align='center', alpha=0.5, ecolor='black')
plt.show(block=True)


print('done')
