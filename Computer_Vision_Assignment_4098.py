import cv2
import numpy as np
import os


os.makedirs("outputs", exist_ok=True)

# Load grayscale and color versions of the image
img_gray = cv2.imread("carpagani.jpg", cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread("carpagani.jpg")

# Error handling
if img_gray is None or img_color is None:
    raise FileNotFoundError("Make sure 'carpagani.jpg' is in the same directory.")

# 1. Reduce intensity levels
def reduce_intensity_levels(img, levels):
    factor = 256 // levels
    reduced_img = (img // factor) * factor
    return reduced_img

for levels in [2, 4, 8, 16, 32, 64, 128]:
    reduced = reduce_intensity_levels(img_gray, levels)
    cv2.imwrite(f"outputs/intensity_{levels}.jpg", reduced)

# 2. Spatial averaging (blurring)
def apply_average_filter(img, kernel_size):
    return cv2.blur(img, (kernel_size, kernel_size))

for k in [3, 10, 20]:
    blurred = apply_average_filter(img_gray, k)
    cv2.imwrite(f"outputs/blur_{k}x{k}.jpg", blurred)

# 3. Image rotation (45 and 90 degrees)
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Compute bounding box to avoid cropping after rotation
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(img, matrix, (new_w, new_h))
    return rotated

for angle in [45, 90]:
    rotated = rotate_image(img_color, angle)
    cv2.imwrite(f"outputs/rotated_{angle}.jpg", rotated)

# 4. Blockwise averaging (non-overlapping)
def blockwise_average(img, block_size):
    h, w = img.shape[:2]
    img_out = img.copy()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if block.shape[0] < block_size or block.shape[1] < block_size:
                continue
            avg = np.mean(block, axis=(0, 1), keepdims=True).astype(np.uint8)
            img_out[i:i+block_size, j:j+block_size] = avg
    return img_out

for b in [3, 5, 7]:
    reduced_img = blockwise_average(img_color, b)
    cv2.imwrite(f"outputs/blockwise_avg_{b}x{b}.jpg", reduced_img)

print("Tasks completed. Please Check the 'outputs' folder.")
