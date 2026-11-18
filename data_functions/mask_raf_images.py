"""
7/7/26
Create versions of the RAF-DB images with masks on either top / bottom halves.
"""
from pathlib import Path
from PIL import Image, ImageDraw
import random
import numpy as np
import matplotlib.pyplot as plt


W = 224
H = 224
N_PX_IMG = W * H
N_PX_MSK = W * (H // 4)
GRID_SIZE = 14
patch_w = W // GRID_SIZE
patch_h = H // GRID_SIZE
N_PTCH_IMG = GRID_SIZE * GRID_SIZE
N_PTCH_MSK = N_PTCH_IMG // 4

assert (N_PTCH_MSK * (W // GRID_SIZE) * (H // GRID_SIZE)) == N_PX_MSK  # num of masked pixels is the same for patch/px.

mask_parts = {
    'eyes': {
        'part': 2,
        'offset': 10,
    },
    'mouth': {
        'part': 4,
        'offset': -15,
    },
    'mouth+eyes': {
        'parts': [2, 4],
        'offsets': [10, -15],
    },
    'nose': {
        'part': 3,
        'offset': 5,
        'adjust_height': -10,
    },
    'forehead': {
        'part': 1,
        'offset': 10,
    },
}

mask_patchs = {
    'eyes': [
        58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109
    ],
    'mouth': [
        143, 144, 145, 146, 147, 148, 149, 150,
        157, 158, 159, 160, 161, 162, 163, 164,
        171, 172, 173, 174, 175, 176, 177, 178,
        185, 186, 187, 188, 189, 190, 191, 192
    ],
    'nose': [
        115, 116, 117, 118, 119, 120, 121, 122,
        129, 130, 131, 132, 133, 134, 135, 136
    ],
    'forehead': [
        17, 18, 19, 20, 21, 22, 23, 24,
        31, 32, 33, 34, 35, 36, 37, 38,
        45, 46, 47, 48, 49, 50, 51, 52
    ],

}
mask_patchs['nose+forehead'] = mask_patchs['nose'] + mask_patchs['forehead']
mask_patchs['mouth+eyes'] = mask_patchs['mouth'] + mask_patchs['eyes']


def show_patch_indices(img, grid_output_pth=None, grid_size=GRID_SIZE, fontsize=8,
                       text_color="black", grid_color="white",
                       alpha=0.8):
    """
    Display an image with ViT patch indices overlaid.

    Parameters
    ----------
    img : str | PIL.Image.Image
        Image path or PIL image.
    grid_size : int
        Number of patches per side (default 14 for ViT-B/16 at 224x224).
    fontsize : int
        Font size of patch numbers.
    text_color : str
        Color of patch indices.
    grid_color : str
        Color of grid lines.
    alpha : float
        Transparency of grid lines.
    """

    if isinstance(img, str):
        img = Image.open(img).convert("RGB")

    width, height = img.size
    patch_w = width / grid_size
    patch_h = height / grid_size

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Draw grid
    for i in range(grid_size + 1):
        ax.axhline(i * patch_h, color=grid_color, lw=0.7, alpha=alpha)
        ax.axvline(i * patch_w, color=grid_color, lw=0.7, alpha=alpha)

    # Draw patch indices
    for row in range(grid_size):
        for col in range(grid_size):
            idx = row * grid_size + col

            x = (col + 0.5) * patch_w
            y = (row + 0.5) * patch_h

            ax.text(
                x, y, str(idx),
                color=text_color,
                fontsize=fontsize,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1)
            )

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if grid_output_pth is not None:
        fig.savefig(grid_output_pth)


def get_ptch_boxes(patch_inds):
    ptch_boxes = []
    for idx in patch_inds:
        ptch_x = idx % GRID_SIZE
        ptch_y = idx // GRID_SIZE
        ptch_boxes.append(
            (
                ptch_x * patch_w,
                ptch_y * patch_h,
                (ptch_x + 1) * patch_w - 1,
                (ptch_y + 1) * patch_h - 1,
            )
        )

    return ptch_boxes


def mask_half_images(
    image_dir,
    parts=(0,),
    offsets=(0,),
    adjust_heights=(0,),
    patch_inds=None,
    mask_color=(0, 0, 0),
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"),
    overwrite=False,
    subset='test',
    seed=None,
    random_exclude=None,
    n_ptch_mask = N_PTCH_MSK,
    mask_mode='rect',
    show_patch_grid=False,
    part_name='',
):
    """
    Copy images from image_dir into a new sibling directory named <original>_masked,
    with either the top or bottom half masked.

    Parameters
    ----------
    image_dir : str or Path
        Path to directory containing images.

    parts : Tuple
        Which quarter of the image to mask, indexing from top to bottom.
        Choose from: (1, 2, 3, 4)

    offsets: Tuple
        for each part in parts, how many pixels to shift upward (positive) / downward (negative)

    adjust_heights: Tuple
        for each part in parts, how many pixels to widen (positive) / narrow (negative)

    patch_inds: List
        which patches to mask.
        Relevant only if mask_mode = 'chosen_patch'.

    mask_color : tuple
        RGB mask color. Default is black.

    image_extensions : tuple
        Image file extensions to process.

    overwrite : bool
        If False, skip output files that already exist.

    subset: 'train' / 'test' / ''

    seed: int
        For setting seed.
        Relevant only if mask_mode = 'random_px' / 'random_patch'.

    mask_mode: 'rect' (then provide parts & offsets) / 'random_px' / 'random_patch' / 'chosen_patch'

    Returns
    -------
    Path
        Path to the created masked-image directory.
    """

    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {image_dir}")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    if 'random' in mask_mode:
        if seed is not None:
            random.seed(seed)
        output_dir = image_dir.with_name(image_dir.name + f"_masked_{mask_mode}_seed{seed}")
        if mask_mode == 'random_px':
            random_px_indices = random.sample(range(N_PX_IMG), N_PX_MSK)
            X = []
            Y = []
            for idx in random_px_indices:
                X.append(idx % W)
                Y.append(idx // W)
        elif mask_mode == 'random_patch':
            exclude_inds = []
            if random_exclude is not None:
                for part in random_exclude:
                    exclude_inds += mask_patchs[part]
            random_ptch_indices = random.sample([i for i in range(N_PTCH_IMG) if i not in exclude_inds], n_ptch_mask)
            ptch_boxes = get_ptch_boxes(random_ptch_indices)
            if (n_ptch_mask != N_PTCH_MSK) | (random_exclude is not None):
                ptch_suf = '' if (n_ptch_mask == N_PTCH_MSK) else f"_{n_ptch_mask}ptchs"
                exc_suf = '' if (random_exclude is None) else f"_outside_{'&'.join(random_exclude)}"
                output_dir = output_dir.with_name(output_dir.name + f"{ptch_suf}{exc_suf}")
    elif mask_mode == 'rect':
        for part in parts:
            if part not in [1, 2, 3, 4]:
                raise ValueError("part must be 1/2/3/4")
        dir_label = f"quarts{'_'.join([str(p) for p in parts])}" \
                    f"_offsets{'_'.join([str(o) for o in offsets])}" \
                    f"_heights{'_'.join([str(h) for h in adjust_heights])}" \
            if len(parts) > 1 else f"quart{parts[0]}_offset{offsets[0]}_height{adjust_heights[0]}"
        output_dir = image_dir.with_name(image_dir.name + f"_masked_{dir_label}")

    elif mask_mode == 'chosen_patch':
        ptch_boxes = get_ptch_boxes(patch_inds)
        output_dir = image_dir.with_name(image_dir.name + f"_masked_patches_{part_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions and p.name.startswith(subset)
    ]

    for img_path in image_files:
        output_path = output_dir / img_path.name

        if show_patch_grid:
            grid_output_dir = image_dir.with_name('show_patch_grid_indexes')
            with Image.open(img_path) as img:
                img = img.copy()
                show_patch_indices(img, grid_output_pth=grid_output_dir/img_path.name)

        if output_path.exists() and not overwrite:
            continue

        with Image.open(img_path) as img:
            img = img.copy()

            width, height = img.size
            qurt_y = height // 4

            if mask_mode == 'random_px':
                img_arr = np.array(img)
                img_arr[X, Y] = mask_color
                img = Image.fromarray(img_arr)
            elif (mask_mode == 'random_patch') | (mask_mode == 'chosen_patch'):
                draw = ImageDraw.Draw(img)
                for box in ptch_boxes:
                    draw.rectangle(
                        box,
                        fill=mask_color,
                    )
            elif mask_mode == 'rect':
                draw = ImageDraw.Draw(img)
                for part, offset, adj_h in zip(parts, offsets, adjust_heights):
                    if part == 1:
                        mask_box = (0, offset-(0 if (adj_h > 0) else (adj_h // 2)), width, qurt_y+offset+adj_h)
                    elif part == 2:
                        mask_box = (0, qurt_y+offset-(adj_h // 2), width, (qurt_y*2)+offset+adj_h)
                    elif part == 3:
                        mask_box = (0, (qurt_y*2)+offset-(adj_h // 2), width, (qurt_y*3)+offset+adj_h)
                    else:
                        mask_box = (0, (qurt_y*3)+offset-(adj_h if (adj_h > 0) else (adj_h // 2)), width, height+offset+adj_h)

                    draw.rectangle(mask_box, fill=mask_color)

            img.save(output_path)

    return output_dir


raf_mean = (124, 116, 103)
part_name = 'nose'
mask_half_images(
    image_dir="/home/projects/bagon/ilanaveh/data/RAF-DB/basic/Image/aligned_224",
    # parts=mask_parts['mouth+eyes']['parts'],
    # parts=[mask_parts['forehead']['part']],
    # parts=[mask_parts[part_name]['part']],
    mask_color=raf_mean,
    # offsets=mask_parts['mouth+eyes']['offsets'],
    # offsets=[mask_parts['forehead']['offset']],
    # offsets=[mask_parts[part_name]['offset']],
    # adjust_heights=[mask_parts[part_name]['adjust_height']],
    subset='',
    seed=2,
    mask_mode='chosen_patch',  # 'rect' / 'chosen_patch' / 'random_px' / 'random_patch'
    patch_inds=mask_patchs[part_name],
    show_patch_grid=False,
    part_name=part_name,
    # n_ptch_mask=40,  # default: N_PTCH_MSK (1/4 of the patches in image)
    # random_exclude=['mouth', 'eyes']
)
