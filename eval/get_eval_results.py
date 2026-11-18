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

save2file = True
append2existing = True
file_name = 'results'
get_reps = False
force_no_pooling_in_test = False

blurs = np.arange(1, 17)
# model_names = [f'RAF_blur{b}_pretrained{pt}{fr_suf}' for b in blurs for pt in ['8-16', '4-8', '0'] for fr_suf in ['_freeze']]
# model_names = [m for m in model_names if ('blur8_pretrained8' not in m) and (m != 'RAF_blur8_pretrained0-8_freeze')]
# model_names = ['RAF_blur8_pretrained8-16_freeze']
# model_names = ['RAF_blur0_pretrained4_freeze']
# model_names = [
    # 'RAF_blur8_mask_eyes_freeze',
    #            'RAF_blur8_mask_mouth_freeze',
    #            'RAF_blur8_mask_eyes_pretrained0-8_freeze',
    #            'RAF_blur8_mask_mouth_pretrained0-8_freeze',
    #            'RAF_blur8_mask_mouth_eyes_freeze',
    #            'RAF_blur8_mask_mouth_eyes_pretrained0-8_freeze',
    #            'RAF_blur8_mask_random_patch_freeze',
    #            'RAF_blur8_mask_random_patch_pretrained0-8_freeze',
    #            'RAF_blur8_mask_random_patch_s2_freeze',
    #            'RAF_blur8_mask_random_patch_s3_freeze',
    #            'RAF_blur8_mask_random_patch_s4_freeze',
    #            'RAF_blur8_mask_random_patch_s2_pretrained0-8_freeze',
    #            'RAF_blur8_mask_random_patch_s3_pretrained0-8_freeze',
    #            'RAF_blur8_mask_random_patch_s4_pretrained0-8_freeze',
    #            'RAF_blur8_mask_nose_freeze',
    #            'RAF_blur8_mask_nose_pretrained0-8_freeze',
    #            'RAF_blur8_mask_forehead_freeze',
    #            'RAF_blur8_mask_forehead_pretrained0-8_freeze',
    #            'RAF_blur8_mask_nose_narrow_freeze',
    #            'RAF_blur8_mask_nose_narrow_pretrained0-8_freeze',
    #            'RAF_blur8_mask_nose-forehead_patches_freeze',
    #            'RAF_blur8_mask_nose-forehead_patches_pretrained0-8_freeze',
    #            'RAF_blur8_mask_eyes_patches_freeze',
    #            'RAF_blur8_mask_eyes_patches_pretrained0-8_freeze',
               # 'RAF_blur8_mask_random_patch_40ptchs_s1_freeze',
               # 'RAF_blur8_mask_random_patch_40ptchs_s1_pretrained0-8_freeze',
               # 'RAF_blur8_mask_random_patch_40ptchs_s2_freeze',
               # 'RAF_blur8_mask_random_patch_40ptchs_s2_pretrained0-8_freeze',
               # 'RAF_blur8_mask_random_patch_40ptchs_s3_freeze',
               # 'RAF_blur8_mask_random_patch_40ptchs_s3_pretrained0-8_freeze',
               # ]
# model_names = ['RAF_blur8_noPool_freeze', 'RAF_blur8_pretrained0-8_noPool_freeze', 'RAF_blur0_noPool_freeze',
#                'RAF_blur8_noPool', 'RAF_blur8_pretrained0-8_noPool', 'RAF_blur0_noPool']
# model_names = ['RAF_blur8_pretrained0-8_freeze_fix_projs']
model_names = ['RAF_blur0', 'RAF_blur8', 'RAF_blur0-8']
test_blurs = {m: None for m in model_names}  # None for using original config.
for mdl in model_names:
    # test_blurs[mdl] = int(mdl.split('blur')[1].split('_')[0])
    test_blurs[mdl] = 6

suf = '_fix_projs'  # uncomment for 'RAF_blur8_pretrained0-8_freeze_fix_projs' model.
# suf = ''  # comment for 'RAF_blur8_pretrained0-8_freeze_fix_projs' model.

if append2existing and os.path.isfile(f"{file_name}.json"):
    with open(f"{file_name}.json", "r") as f:
        results = json.load(f)
    for mdl in model_names:
        if mdl not in results.keys():
            results[mdl] = {'rep0': {}}
else:
    results = {m: {'rep0': {}} for m in model_names}

for mdl in model_names:
    cfg_pth = f'../configs/apvit/{mdl.replace(suf, "")}.py'
    cfg = mmcv.Config.fromfile(cfg_pth)

    print(mdl)
    if not force_no_pooling_in_test and (f'test_{test_blurs[mdl]}' in results[mdl]['rep0']):
        continue

    if force_no_pooling_in_test and (f'test_{test_blurs[mdl]}_testNoPool' in results[mdl]['rep0']):
        continue

    if os.path.isfile(f'../work_dirs/{mdl}/epoch_40.pth'):
        test_args = [cfg_pth, f'../work_dirs/{mdl}/epoch_40.pth', '--return_acc']

        if test_blurs[mdl] is not None:
            test_blur = test_blurs[mdl]
            test_args.extend(['--test_blur', str(test_blur)])
        else:
            test_blur = cfg.data['test']['pipeline'][1]['sigma_min'] if cfg.data['test']['pipeline'][1][
                                                                            'type'] == 'GaussianBlur' else 0
        if force_no_pooling_in_test:
            test_args.append('--force_no_pool')

        print(test_args)
        cur_results = test.main(test_args)

        if force_no_pooling_in_test:
            results[mdl]['rep0'][f'test_{test_blur}_testNoPool'] = float(np.round(cur_results['top-1'], 2))
        else:
            results[mdl]['rep0'][f'test_{test_blur}'] = float(np.round(cur_results['top-1'], 2))

        if save2file:
            with open(f"{file_name}.json", "w") as f:
                json.dump(results, f, indent=1)
    else:
        print(f"Training of model {mdl} hasn't finished. Moving to next model...")

    # Add repetitions if exist:
    if get_reps:
        for r in range(1, 5):
            m_rep_file = mdl + f'_rep{r}' if r > 1 else mdl + f'_rep'
            if os.path.isdir(f'../work_dirs/{m_rep_file}'):
                rep_str = f"rep{r}"
                if rep_str not in results[mdl]:
                    results[mdl][rep_str] = {}
                print(f"{mdl}, {rep_str}")
                print(f"Testing with blur: {test_blur}")
                if test_blurs[mdl] is not None:
                    cur_results = test.main(
                        [cfg_pth, f'../work_dirs/{m_rep_file}/epoch_40.pth', '--return_acc', '--test_blur',
                         str(test_blur)])
                else:
                    cur_results = test.main([cfg_pth, f'../work_dirs/{m_rep_file}/epoch_40.pth', '--return_acc'])

                results[mdl][rep_str][f'test_{test_blur}'] = float(np.round(cur_results['top-1'], 2))

                if save2file:
                    with open(f"{file_name}.json", "w") as f:
                        json.dump(results, f, indent=1)
