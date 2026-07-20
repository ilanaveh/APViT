"""
7/7/26
Create versions of the RAF-DB images with masks on either top / bottom halves.
"""
from pathlib import Path
from PIL import Image, ImageDraw


def mask_half_images(
    image_dir,
    part=1,
    offset=0,
    mask_color=(0, 0, 0),
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"),
    overwrite=True,
):
    """
    Copy images from image_dir into a new sibling directory named <original>_masked,
    with either the top or bottom half masked.

    Parameters
    ----------
    image_dir : str or Path
        Path to directory containing images.

    part : [1, 2, 3, 4]
        Which quarter of the image to mask, indexing from top to bottom.

    mask_color : tuple
        RGB mask color. Default is black.

    image_extensions : tuple
        Image file extensions to process.

    overwrite : bool
        If False, skip output files that already exist.

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

    if part not in [1, 2, 3, 4]:
        raise ValueError("half must be 1/2/3/4")

    output_dir = image_dir.with_name(image_dir.name + f"_masked_quart{part}_offset{offset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions and p.name.startswith('train')
    ]

    for img_path in image_files:
        output_path = output_dir / img_path.name

        if output_path.exists() and not overwrite:
            continue

        with Image.open(img_path) as img:
            img = img.copy()

            width, height = img.size
            qurt_y = height // 4

            draw = ImageDraw.Draw(img)

            if part == 1:
                mask_box = (0, 0, width, qurt_y+offset)
            elif part == 2:
                mask_box = (0, qurt_y+offset, width, (qurt_y*2)+offset)
            elif part == 3:
                mask_box = (0, (qurt_y*2)+offset, width, (qurt_y*3)+offset)
            else:
                mask_box = (0, (qurt_y*3)+offset, width, height+offset)

            draw.rectangle(mask_box, fill=mask_color)

            img.save(output_path)

    return output_dir


raf_mean = (124, 116, 103)
mask_half_images(
    image_dir="/home/projects/bagon/ilanaveh/data/RAF-DB/basic/Image/aligned_224",
    part=4,
    mask_color=raf_mean,
    offset=-15,
)
