from tools import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


model_name = 'RAF_blur0'

results = test.main(['configs/apvit/RAF_terminal_deit.py', f'work_dirs/{model_name}/epoch_40.pth', '--return_labels'])
# results = test.main(['configs/apvit/RAF_blur8.py', f'work_dirs/{model_name}/epoch_40.pth', '--return_labels'])
CLASSES = np.unique(results['target_class'])

breakdown = {c_tar: {c_pred: 0 for c_pred in CLASSES} for c_tar in CLASSES}
breakdown_percent = {c_tar: {c_pred: 0 for c_pred in CLASSES} for c_tar in CLASSES}
total_samples = {c: 0 for c in CLASSES}

for tar, pred in zip(results['target_class'], results['pred_class']):
    breakdown[tar][pred] += 1
    total_samples[tar] += 1

assert np.sum(list(total_samples.values())) == 3068  # number of samples in RAF-DB test set.

# Convert to percent (out of total samples of each class):
for tar in CLASSES:
    for pred in CLASSES:
        breakdown_percent[tar][pred] = breakdown[tar][pred] / total_samples[tar] * 100

conf_matrix = pd.DataFrame(breakdown_percent).T  # .T transposes so rows = true labels, cols = predicted

# Sort rows & columns by # samples of each class:
order = sorted(conf_matrix.index, key=lambda x: total_samples[x], reverse=True)
conf_matrix = conf_matrix.reindex(index=order, columns=order)

# Add # samples of each class to row labels:
conf_matrix = conf_matrix.rename(index={c: f'{c} ({total_samples[c]})' for c in conf_matrix.index})

# plot
ttl = f"Model {model_name}"
f = plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(ttl)
ax = plt.gca()
ax.set_yticklabels([str(x).replace(' ', '\n') for x in conf_matrix.index])
print()
