import argparse
import os
from Joonkyo.src.palette_extractor import extract_colors, save_colors_to_json
from Joonkyo.src.palette_image import save_palette_image


def print_colors(colors_rgb, colors_hex):
    print("=== Extracted Colors ===")
    for i, (rgb, hx) in enumerate(zip(colors_rgb, colors_hex), start=1):
        print(f"{i}: RGB={rgb}, HEX={hx}")


def ensure_directory(path):
    """Ensure the parent directory of the given file path exists."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="ColorPaletteGen")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--k", type=int, default=5, help="number of colors to extract")
    parser.add_argument("--output", default="palette.png", help="output palette image path")
    parser.add_argument("--json", default=None, help="json output path")
    return parser.parse_args()


def main():
    args = parse_args()

    # 색상 추출
    colors_rgb, colors_hex = extract_colors(args.image, args.k)
    print_colors(colors_rgb, colors_hex)

    # 출력 폴더 자동 생성
    ensure_directory(args.output)

    # 팔레트 저장
    save_palette_image(colors_rgb, args.output)
    print(f"Palette saved to: {args.output}")

    # JSON 저장 옵션
    if args.json:
        ensure_directory(args.json)
        save_colors_to_json(colors_rgb, colors_hex, args.json)
        print(f"JSON saved to: {args.json}")


if __name__ == "__main__":
    main()
