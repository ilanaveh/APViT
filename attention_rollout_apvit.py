"""
28/07/26
Implementation of "Quantifying Attention Flow in Transformers" (Abner & Zuidema, 2020).
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from mmcls.models.builder import build_classifier
from mmcls.datasets.pipelines import Compose
import argparse
import mmcv
from mmcv.runner import load_checkpoint
import timm


class APViTAttentionRollout:
    """
    Class-agnostic Attention Rollout for APViT-like models.

    The class registers forward-pre-hooks on every transformer attention
    module and reconstructs its post-softmax attention matrix.

    Expected attention-module attributes:
        qkv
        num_heads
        scale

    Standard APViT location:
        model.backbone.vit.blocks[i].attn
    or, depending on the wrapper:
        model.vit.blocks[i].attn
        model.blocks[i].attn
    """

    def __init__(
        self,
        model: nn.Module,
        vit: nn.Module = None,
        head_fusion: str = "mean",
        discard_ratio: float = 0.0,
    ) -> None:
        if head_fusion not in {"mean", "max", "min"}:
            raise ValueError(
                "head_fusion must be 'mean', 'max', or 'min'."
            )

        if not 0.0 <= discard_ratio < 1.0:
            raise ValueError(
                "discard_ratio must satisfy 0 <= discard_ratio < 1."
            )

        self.model = model
        self.vit = vit if vit is not None else self._find_vit(model)
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        self.attentions: list[Tensor] = []
        self.handles: list[Any] = []

        self._register_hooks()

    @staticmethod
    def _find_vit(model: nn.Module) -> nn.Module:
        """
        Locate the ViT component in common APViT wrappers.
        """
        candidates = [
            model,
            getattr(model, "vit", None),
            getattr(model, "transformer", None),
            getattr(model, "backbone", None),
        ]

        backbone = getattr(model, "backbone", None)

        if backbone is not None:
            candidates.extend(
                [
                    getattr(backbone, "vit", None),
                    getattr(backbone, "transformer", None),
                    getattr(backbone, "backbone", None),
                ]
            )

        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "blocks"):
                blocks = getattr(candidate, "blocks")
                if len(blocks) > 0 and hasattr(blocks[0], "attn"):
                    return candidate

        raise AttributeError(
            "Could not locate the APViT transformer blocks. "
            "Pass the transformer explicitly, for example:\n"
            "    APViTAttentionRollout(model, vit=model.backbone.vit)"
        )

    def _register_hooks(self) -> None:
        for block_index, block in enumerate(self.vit.blocks):
            attention_module = block.attn

            required_attributes = ("qkv", "num_heads", "scale")

            if not all(
                hasattr(attention_module, name)
                for name in required_attributes
            ):
                raise AttributeError(
                    f"Attention module in block {block_index} does not "
                    f"have all required attributes: {required_attributes}"
                )

            handle = attention_module.register_forward_pre_hook(
                self._make_attention_hook(block_index)
            )
            self.handles.append(handle)

    def _make_attention_hook(
        self,
        block_index: int,
    ):
        """
        Reconstruct the attention matrix from the input to Attention.forward.
        """

        def hook(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            if len(inputs) == 0:
                raise RuntimeError(
                    f"No input was received by attention block {block_index}."
                )

            x = inputs[0]

            if x.ndim != 3:
                raise RuntimeError(
                    f"Expected attention input [B, N, C], got "
                    f"{tuple(x.shape)} in block {block_index}."
                )

            batch_size, num_tokens, embedding_dim = x.shape
            num_heads = module.num_heads

            if embedding_dim % num_heads != 0:
                raise RuntimeError(
                    f"Embedding dimension {embedding_dim} is not divisible "
                    f"by {num_heads} heads."
                )

            head_dim = embedding_dim // num_heads

            # Same calculation as APViT Attention.forward().
            qkv = module.qkv(x)
            qkv = qkv.reshape(
                batch_size,
                num_tokens,
                3,
                num_heads,
                head_dim,
            )

            # APViT applies head dropout before splitting q, k, and v.
            # In eval mode this should be inactive.
            if hasattr(module, "head_drop"):
                qkv = module.head_drop(qkv)

            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]

            attention = (q @ k.transpose(-2, -1)) * module.scale
            attention = attention.softmax(dim=-1)

            # Store on CPU to reduce GPU memory usage.
            self.attentions.append(attention.detach().cpu())

        return hook

    def clear(self) -> None:
        self.attentions.clear()

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    def __enter__(self) -> "APViTAttentionRollout":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove_hooks()

    def _fuse_heads(self, attention: Tensor) -> Tensor:
        """
        Input:
            [B, heads, tokens, tokens]

        Output:
            [B, tokens, tokens]
        """
        if self.head_fusion == "mean":
            fused = attention.mean(dim=1)
        elif self.head_fusion == "max":
            fused = attention.max(dim=1).values
        else:
            fused = attention.min(dim=1).values

        return fused

    def _discard_low_attention(self, attention: Tensor) -> Tensor:
        """
        Optionally zero the lowest attention values in each layer.

        CLS-related entries are retained.
        """
        if self.discard_ratio == 0:
            return attention

        batch_size, num_tokens, _ = attention.shape

        flat = attention.reshape(batch_size, -1)
        num_discard = int(flat.shape[-1] * self.discard_ratio)

        if num_discard == 0:
            return attention

        _, indices = flat.topk(
            num_discard,
            dim=-1,
            largest=False,
        )

        discarded = flat.clone()
        discarded.scatter_(dim=-1, index=indices, value=0.0)
        discarded = discarded.reshape(
            batch_size,
            num_tokens,
            num_tokens,
        )

        # Do not discard CLS row or column.
        discarded[:, 0, :] = attention[:, 0, :]
        discarded[:, :, 0] = attention[:, :, 0]

        return discarded

    def calculate_rollout(self) -> Tensor:
        """
        Returns:
            Tensor of shape [B, number_of_spatial_tokens].

        The returned values are CLS-to-token rollout scores.
        """
        if not self.attentions:
            raise RuntimeError(
                "No attention matrices were captured. "
                "Run a forward pass before calculate_rollout()."
            )

        token_counts = {
            attention.shape[-1]
            for attention in self.attentions
        }

        if len(token_counts) != 1:
            raise RuntimeError(
                "The number of tokens changes between transformer blocks: "
                f"{sorted(token_counts)}.\n"
                "Ordinary matrix-product rollout cannot be applied directly "
                "when APViT removes tokens between blocks. Token-selection "
                "indices must first be mapped back to the original grid."
            )

        first_attention = self.attentions[0]
        batch_size = first_attention.shape[0]
        num_tokens = first_attention.shape[-1]

        rollout = torch.eye(
            num_tokens,
            dtype=first_attention.dtype,
        ).unsqueeze(0).repeat(batch_size, 1, 1)

        identity = torch.eye(
            num_tokens,
            dtype=first_attention.dtype,
        ).unsqueeze(0)

        for attention in self.attentions:
            fused = self._fuse_heads(attention)
            fused = self._discard_low_attention(fused)

            # Account for the residual connection.
            augmented = fused + identity

            # Row-normalize.
            augmented = augmented / augmented.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)

            # Compose information flow across layers.
            rollout = augmented @ rollout

        # Row 0 is the CLS query; column 0 is the CLS token itself.
        return rollout[:, 0, 1:]

    @torch.no_grad()
    def __call__(
        self,
        model_input: Tensor,
        forward_function=None,
    ) -> Tensor:
        """
        Run the model and return rollout scores.

        Use forward_function when the model requires something other than:
            model(model_input)
        """
        self.clear()
        self.model.eval()

        if forward_function is None:
            _ = self.model(model_input, return_loss=False)
        else:
            _ = forward_function(model_input)

        return self.calculate_rollout()


def rollout_to_spatial_map(
    rollout: Tensor,
    feature_grid: tuple[int, int] = None,
) -> Tensor:
    """
    Convert [B, N] rollout scores into [B, H, W].

    Parameters
    ----------
    rollout:
        CLS-to-spatial-token rollout, shape [B, N].

    feature_grid:
        Spatial dimensions of the tokens entering the transformer.

        Example:
            (14, 14) for 196 tokens
            (7, 7) for 49 tokens
    """
    if rollout.ndim != 2:
        raise ValueError(
            f"Expected rollout [B, N], got {tuple(rollout.shape)}."
        )

    batch_size, num_spatial_tokens = rollout.shape

    if feature_grid is None:
        side = math.isqrt(num_spatial_tokens)

        if side * side != num_spatial_tokens:
            raise ValueError(
                f"{num_spatial_tokens} tokens do not form a square grid. "
                "Pass feature_grid=(height, width)."
            )

        feature_grid = (side, side)

    grid_height, grid_width = feature_grid

    if grid_height * grid_width != num_spatial_tokens:
        raise ValueError(
            f"feature_grid={feature_grid} contains "
            f"{grid_height * grid_width} locations, but rollout contains "
            f"{num_spatial_tokens} spatial tokens."
        )

    spatial_map = rollout.reshape(
        batch_size,
        grid_height,
        grid_width,
    )

    minimum = spatial_map.amin(
        dim=(-2, -1),
        keepdim=True,
    )
    maximum = spatial_map.amax(
        dim=(-2, -1),
        keepdim=True,
    )

    spatial_map = (spatial_map - minimum) / (
        maximum - minimum
    ).clamp_min(1e-12)

    return spatial_map


def parse_args():
    parser = argparse.ArgumentParser(description='Vizualize APViT model CNN attention')
    parser.add_argument('--config', type=str, default="configs/apvit/RAF_run_from_terminal.py",
                        help='config file path'
                             'The default is for official model with vit (8 blocks), '
                             'if using local models (deit, 12 blocks) change to: configs/apvit/RAF_terminal_deit.py')
    parser.add_argument('--keep_num', type=int, default=196, help='')  # currently, attention-rollout works only on the full set of pathe
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
    cfg.model.vit.vit_pool_configs['keep_rates'] = [1.] * 8  # currently, rollout method ca
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
    # if args.blur:
    #     trans_list = [dict(type='GaussianBlur', sigma_min=args.blur, sigma_max=args.blur)] + trans_list

    test_preprocess = Compose(trans_list)

    img = mmcv.imread(args.img_pth)

    # preprocess the image
    data = test_preprocess(dict(img=img))
    data['img'] = data['img'][None, ...].cuda()

    with APViTAttentionRollout(
        model=classifier,
        head_fusion="mean",
        discard_ratio=0.0,
    ) as rollout_extractor:

        rollout = rollout_extractor(data['img'])

    print("Rollout vector shape:", rollout.shape)

    # Replace this with the true CNN feature-map size entering your ViT.
    # For example, use (14, 14) only when there are 196 spatial tokens.
    rollout_map = rollout_to_spatial_map(
        rollout,
        feature_grid=(14, 14),
    )


if __name__ == '__main__':
    main()
