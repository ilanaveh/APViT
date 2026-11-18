"""
29/10/26
Based on get_eval_results.py. But get the balanced accuracy, by averaging class-wise accuracies.
"""
import mmcv
import json
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import test

save2file = True
append2existing = True
file_name = 'results_balanced'
force_no_pooling_in_test = False

# how many images from each class [Angry, Disgust, Fear, Sad, Happy, Surprise, Neutral]:
class_n = [162, 160, 74, 478, 1185, 329, 680]

# blurs = np.arange(17)
blurs = [12, 24]
pt_blurs = ['0', '0-8', '8']
model_names = [f'RAF_blur{bl}_pretrained{pt_bl}_freeze' for bl in blurs for pt_bl in pt_blurs]

suf = '_fix_projs'  # uncomment for 'RAF_blur8_pretrained0-8_freeze_fix_projs' model.
# suf = ''  # comment for 'RAF_blur8_pretrained0-8_freeze_fix_projs' model.

if 'RAF_blur8_pretrained0-8_freeze' in model_names:
    model_names[model_names.index('RAF_blur8_pretrained0-8_freeze')] += suf

for i, mdl in enumerate(model_names):
    if mdl.split('blur')[1].split('_')[0] == mdl.split('pretrained')[1].split('_')[0]:
        model_names[i] = mdl.split('pretrained')[0] + 'freeze'

test_blurs = {m: None for m in model_names}  # None for using original config.
for mdl in model_names:
    test_blurs[mdl] = int(mdl.split('blur')[1].split('_')[0])

if append2existing and os.path.isfile(f"{file_name}.json"):
    with open(f"{file_name}.json", "r") as f:
        results = json.load(f)
    for mdl in model_names:
        if mdl not in results.keys():
            results[mdl] = {}
else:
    results = {m: {} for m in model_names}

for mdl in model_names:
    cfg_pth = f'../configs/apvit/{mdl.replace(suf, "")}.py'
    cfg = mmcv.Config.fromfile(cfg_pth)

    print(mdl)
    if f'test_{test_blurs[mdl]}' in results[mdl]:
        continue

    if os.path.isfile(f'../work_dirs/{mdl}/epoch_40.pth'):
        test_args = [cfg_pth, f'../work_dirs/{mdl}/epoch_40.pth', '--return_labels']

        if test_blurs[mdl] is not None:
            test_blur = test_blurs[mdl]
            test_args.extend(['--test_blur', str(test_blur)])
        else:
            test_blur = cfg.data['test']['pipeline'][1]['sigma_min'] if cfg.data['test']['pipeline'][1][
                                                                            'type'] == 'GaussianBlur' else 0

        print(test_args)
        cur_results = test.main(test_args)

        assert [list(cur_results['target_label']).count(i) for i in range(7)] == class_n

        class_accs = np.array([np.sum([prd == tru for prd, tru in
                                       zip(cur_results['pred_label'], cur_results['target_label'])
                                       if tru == i]) / class_n[i] for i in range(7)])

        balanced_sem = np.sqrt(np.sum(class_accs * (1 - class_accs) / class_n)) / len(class_accs)

        mean_acc = class_accs.mean() * 100
        sem = balanced_sem * 100

        results[mdl][f'test_{test_blur}_acc'] = float(np.round(mean_acc, 2))
        results[mdl][f'test_{test_blur}_sem'] = float(np.round(sem, 2))

        if save2file:
            with open(f"{file_name}.json", "w") as f:
                json.dump(results, f, indent=1)
    else:
        print(f"Training of model {mdl} hasn't finished. Moving to next model...")

