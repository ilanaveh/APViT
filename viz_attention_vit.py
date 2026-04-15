"""
26/1/26
Based on viz_attention_cnn.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import mmcv
from mmcv.runner import load_checkpoint
from mmcls.models.builder import build_classifier
from mmcls.datasets.pipelines import Compose
from mmcls.models.utils import top_pool


def parse_args():
    parser = argparse.ArgumentParser(description='Vizualize APViT model CNN attention')
    parser.add_argument('--config', type=str, default="configs/apvit/RAF_run_from_terminal.py",
                        help='config file path'
                             'The default is for official model with vit (8 blocks), '
                             'if using local models (deit, 12 blocks) change to: configs/apvit/RAF_terminal_deit.py')
    parser.add_argument('--keep_num', type=int, default=160, help='')
    parser.add_argument('--img_pth', type=str, default='resources/demo.jpg',
                        help="Image for attention visualization."
                             "Options:"
                             "resources/demo.jpg"
                             "resources/test_0002_112.jpg")
    parser.add_argument('--checkpoint_dir', type=str, default='work_dirs',
                        help="Options:"
                             "weights - original pretrained"
                             "work_dirs - locally trained")
    parser.add_argument('--checkpoint_file_name', type=str, default='epoch_40.pth',
                        help="Options:"
                             "APViT_RAF-3eeecf7d.pth - official pretrained"
                             "epoch_40.pth - locally trained")
    parser.add_argument('--checkpoint_model', type=str, default='RAF_blur0',
                        help="Options:"
                             "None - official pretrained"
                             "RAF_blurX - locally trained (X = 0/4/8...")
    parser.add_argument('--blur', type=int, default=None)
    parser.add_argument('--show_im_with_blur', action='store_true')
    parser.add_argument('--not_show_attn', action='store_true')
    parser.add_argument('--show_only_img', action='store_true',
                        help='just plot the original image (with or without blur, according to show_im_with_blur)')
    parser.add_argument('--show_patch_grid', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.extractor.pretrained = None
    cfg.model.vit.pretrained = None
    cfg.model.vit.cnn_pool_config['keep_num'] = args.keep_num
    # cfg.model.vit.vit_pool_configs['keep_rates'] = [1.] * 4 + [0.9] * 4
    # cfg.model.vit.attn_before_proj = True

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)
    cp_full_pth = f'{args.checkpoint_dir}/{args.checkpoint_model}/{args.checkpoint_file_name}' if args.checkpoint_model \
        else f'{args.checkpoint_dir}/{args.checkpoint_file_name}'
    load_checkpoint(classifier, cp_full_pth, map_location='cpu')
    classifier = classifier.to("cuda")
    classifier.eval()

    # define the preprocess for test
    trans_list = [
        dict(type='Resize', size=112),
        dict(type='Normalize',
             mean=[123.675, 116.28, 103.53],
             std=[58.395, 57.12, 57.375]),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ]
    if args.blur:
        trans_list = [dict(type='GaussianBlur', sigma_min=args.blur, sigma_max=args.blur)] + trans_list

    test_preprocess = Compose(trans_list)

    img = mmcv.imread(args.img_pth)

    # preprocess the image
    data = test_preprocess(dict(img=img))
    data['img'] = data['img'][None, ...].cuda()

    if args.show_only_img:
        f, ax = plt.subplots()
        if args.show_im_with_blur and args.blur:
            show_preprocess = Compose([dict(type='GaussianBlur', sigma_min=args.blur, sigma_max=args.blur)])
            img2show = show_preprocess(dict(img=img))['img'][:, :, ::-1]
        else:
            img2show = img.copy()[:, :, ::-1]
        ax.imshow(img2show)
        if args.show_patch_grid:
            ax.set_xticks(np.arange(8, 112, 8))
            ax.set_yticks(np.arange(8, 112, 8))
            ax.grid('on')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show(block=True)
        return

    # Get vit attention-maps per block:
    vit_attn_weights = []
    vit_keep_inds = []

    # To enable following hooks, I added 'self.keep_index' and 'self.attn_weight' properties to PoolingAttention
    # (implemented in APViT.mmcls.models.backbones.modules.vit_pooling.PoolingAttention.forward)

    def hook_attn_fn(module, input, output):
        vit_attn_weights.append(module.attn_weight.cpu())

    def hook_keep_ind_fn(module, input, output):
        if module.keep_index is not None:
            vit_keep_inds.append(module.keep_index.cpu())
        else:
            vit_keep_inds.append(module.keep_index)

    for block in classifier.vit.blocks:
        block.attn.register_forward_hook(hook_attn_fn)
        block.attn.register_forward_hook(hook_keep_ind_fn)

    # First, extract CNN attention, for consistency with indexes (vit indexes are selected only from remains from cnn):
    cnn_attn_map = classifier.extract_attn_map(data['img'])
    cnn_attn_weight = cnn_attn_map.flatten(2).transpose(2, 1)
    if cfg.model.vit.cnn_pool_config is not None:
        cnn_keep_indexes = top_pool(cnn_attn_weight, dim=1, **cfg.model.vit.cnn_pool_config).squeeze().cpu()

    # Forward pass:
    # out = classifier(**data, return_loss=False)

    # Put attention_weight values in correct positions (in original 196 vector)
    current_keep_inds = cnn_keep_indexes
    f, axs = plt.subplots(np.int32(np.ceil(cfg.model.vit.depth / 4)), 4)
    f.set_size_inches((8.8, cfg.model.vit.depth/1.75))  # used to be h=4.8 (for 2 rows).
    axs = axs.flatten()
    for blk in range(cfg.model.vit.depth):
        ax = axs[blk]
        full_attn_weight = np.zeros(196)
        if blk <= 4:  # First five blocks - no need to remove additional patches:
            full_attn_weight[current_keep_inds] = vit_attn_weights[blk].detach().squeeze()[1:]  # remove cls token [index 0]
        else:  # Last three blocks - need to remove additional patches:
            current_keep_inds = current_keep_inds[vit_keep_inds[blk - 1].detach().squeeze()[1:] - 1]  # -1 since 1st was cls token.
            full_attn_weight[current_keep_inds] = vit_attn_weights[blk].detach().squeeze()[1:]
        attn_map = full_attn_weight.reshape([14, 14])
        att_112 = np.kron(attn_map, np.ones((8, 8), dtype=int))  # duplicate each attention value to all pixels in patch

        mask = (att_112 == 0)   # for creating masked_attn - so color map takes into account only the relevant patches.
        masked_att = np.ma.masked_where(mask, att_112)

        if args.show_im_with_blur and args.blur:
            show_preprocess = Compose([dict(type='GaussianBlur', sigma_min=args.blur, sigma_max=args.blur)])
            img_masked = show_preprocess(dict(img=img))['img'][:, :, ::-1]
        else:
            img_masked = img.copy()[:, :, ::-1]
        img_masked[~att_112.astype(bool)] = 255  # mask patches that were already removed by CNN (in white)

        ax.imshow(img_masked)
        if not args.not_show_attn:
            ax.imshow(masked_att, cmap='jet', alpha=.4)

        ax.set_title(f'Block {blk}')
        ax.set_xticks([])
        ax.set_yticks([])
    ttl = args.checkpoint_model if args.checkpoint_model else 'Official'
    f.suptitle(ttl)

    plt.show(block=True)


if __name__ == '__main__':
    main()
