import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

ColorRGB = Tuple[int, int, int]
ColorHEX = str

def rgb_to_hex(rgb: ColorRGB) -> ColorHEX:
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def load_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def reshape_image_to_pixels(image: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    pixels = image.reshape(-1, 3)
    return pixels

def extract_colors(image_path: str, k: int = 5, random_state: int = 42):
    if k <= 0:
        raise ValueError("k must be positive")

    image = load_image(image_path)
    pixels = reshape_image_to_pixels(image)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_

    colors_rgb = [(int(r), int(g), int(b)) for r, g, b in centers]
    colors_hex = [rgb_to_hex(c) for c in colors_rgb]

    return colors_rgb, colors_hex

def save_colors_to_json(colors_rgb, colors_hex, output_path):
    data = {
        "colors": [
            {"index": i + 1, "rgb": list(rgb), "hex": hex_code}
            for i, (rgb, hex_code) in enumerate(zip(colors_rgb, colors_hex))
        ]
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
