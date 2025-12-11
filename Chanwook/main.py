import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

input_path = "examples/input/campus_trees.jpg.png"


def main():
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {input_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    height, width = rgb.shape[:2]

    sky_mask = cv2.inRange(
        hsv,
        (0, 0, 190), 
        (179, 70, 255)
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_abs = np.absolute(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_mask = cv2.threshold(lap_norm, 35, 255, cv2.THRESH_BINARY)

    green_mask = cv2.inRange(
        hsv,
        (25, 40, 40),
        (90, 255, 255)
    )
    brown_mask = cv2.inRange(
        hsv,
        (5, 30, 20),
        (30, 255, 200)
    )
    flower_mask = cv2.inRange(
        hsv,
        (0, 5, 200),       
        (179, 130, 255)
    )

    color_mask = cv2.bitwise_or(green_mask, brown_mask)
    color_mask = cv2.bitwise_or(color_mask, flower_mask)

    sky_inv = cv2.bitwise_not(sky_mask)
    candidate = cv2.bitwise_and(texture_mask, color_mask)
    candidate = cv2.bitwise_and(candidate, sky_inv)

    row_mask = np.zeros_like(candidate)
    row_mask[0:int(height * 0.8), :] = 255
    candidate = cv2.bitwise_and(candidate, row_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)


    gray_bg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB)

    inv_mask = cv2.bitwise_not(mask_bin)
    bg = cv2.bitwise_and(gray_bg, gray_bg, mask=inv_mask)
    fg = cv2.bitwise_and(rgb, rgb, mask=mask_bin)
    result = cv2.add(bg, fg)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask (Tree + Blossoms)")
    plt.imshow(mask_bin, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Tree Segmentation")
    plt.imshow(result)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


    os.makedirs("images", exist_ok=True)

    cv2.imwrite("images/original_campus_trees.png", img)

    cv2.imwrite("images/mask_campus_trees.png", mask_bin)

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/segmentation_campus_trees.png", result_bgr)

if __name__ == "__main__":
    main()
