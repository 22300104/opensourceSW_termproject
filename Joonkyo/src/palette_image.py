from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw

ColorRGB = Tuple[int, int, int]

def save_palette_image(colors_rgb: List[ColorRGB], output_path: str, block_size: int = 100):
    if not colors_rgb:
        raise ValueError("colors_rgb cannot be empty")

    num_colors = len(colors_rgb)
    width = num_colors * block_size
    height = block_size

    palette = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(palette)

    for i, color in enumerate(colors_rgb):
        x0 = i * block_size
        draw.rectangle([x0, 0, x0 + block_size, height], fill=color)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette.save(path)
