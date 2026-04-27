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


save2file = False
append2existing = True
file_name = 'results2'

model_names = ['RAF_blur0', 'RAF_blur8', 'RAF_blur0-8']
test_blurs = {m: None for m in model_names}  # None for using original config.
test_blurs['RAF_blur0'] = 8
test_blurs['RAF_blur8'] = 0
test_blurs['RAF_blur0-8'] = 0

if append2existing and os.path.isfile(f"{file_name}.json"):
    with open(f"{file_name}.json", "r") as f:
        results = json.load(f)
else:
    results = {m: {'rep0': {}} for m in model_names}

for m in model_names:
    cfg_pth = f'../configs/apvit/{m}.py'
    cfg = mmcv.Config.fromfile(cfg_pth)

    print(m)

    if test_blurs[m] is not None:
        test_blur = test_blurs[m]
        print(f"Testing with blur: {test_blur}")
        cur_results = test.main([cfg_pth, f'../work_dirs/{m}/epoch_40.pth', '--return_acc', '--test_blur', str(test_blur)])

    else:
        test_blur = cfg.data['test']['pipeline'][1]['sigma_min'] if cfg.data['test']['pipeline'][1]['type'] == 'GaussianBlur' else 0
        print(f"Testing with blur: {test_blur}")
        cur_results = test.main([cfg_pth, f'../work_dirs/{m}/epoch_40.pth', '--return_acc'])

    results[m]['rep0'][f'test_{test_blur}'] = np.round(cur_results['top-1'], 2)

    # Add repetitions if exist:
    for r in range(1, 5):
        m_rep_file = m + f'_rep{r}' if r > 1 else m + f'_rep'
        if os.path.isdir(f'../work_dirs/{m_rep_file}'):
            rep_str = f"rep{r}"
            if rep_str not in results[m]:
                results[m][rep_str] = {}
            print(f"{m}, {rep_str}")
            print(f"Testing with blur: {test_blur}")
            if test_blurs[m] is not None:
                cur_results = test.main([cfg_pth, f'../work_dirs/{m_rep_file}/epoch_40.pth', '--return_acc', '--test_blur', str(test_blur)])
            else:
                cur_results = test.main([cfg_pth, f'../work_dirs/{m_rep_file}/epoch_40.pth', '--return_acc'])

            results[m][rep_str][f'test_{test_blur}'] = np.round(cur_results['top-1'], 2)

if save2file:
    with open(f"{file_name}.json", "w") as f:
        json.dump(results, f, indent=1)
