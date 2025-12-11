import cv2 as cv
import numpy as np
import sys
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Pyramid vs Direct blending (OpenCV)")
    parser.add_argument("--img1", type=str, required=True, help="Path to first image (e.g., apple.jpg)")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image (e.g., orange.jpg)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    return parser.parse_args()

args = parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Read images
A = cv.imread(args.img1)
B = cv.imread(args.img2)

if A is None or B is None:
    sys.exit("Could not read one of the images.")

# Resize B to match A if needed
if A.shape != B.shape:
    B = cv.resize(B, (A.shape[1], A.shape[0]))

# Generate Gaussian pyramid for A
G = np.float32(A.copy())
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)

# Generate Gaussian pyramid for B
G = np.float32(B.copy())
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)

# Generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i - 1], GE)
    lpA.append(L)

# Generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv.pyrUp(gpB[i])
    L = cv.subtract(gpB[i - 1], GE)
    lpB.append(L)

# Add left and right halves of images
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, :cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

# Reconstruct
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, LS[i])

# Direct blending
rows, cols, _ = A.shape
direct = np.hstack((A[:, :cols // 2], B[:, cols // 2:]))

# Save results
pyramid_path = os.path.join(args.output_dir, "Pyramid_blending.jpg")
direct_path = os.path.join(args.output_dir, "Direct_blending.jpg")

cv.imwrite(pyramid_path, ls_)
cv.imwrite(direct_path, direct)

print(f"Saved pyramid blending result to: {pyramid_path}")
print(f"Saved direct blending result to: {direct_path}")

# Show windows
cv.imshow("Pyramid blending", ls_ / 255.0)
cv.imshow("Direct blending", direct)
cv.waitKey(0)
cv.destroyAllWindows()
