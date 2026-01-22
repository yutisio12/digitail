import cv2
import numpy as np
from preprocess import preprocess_image
from utils import is_valid_digit_contour

IMAGE_PATH = "src/samples/gambar1.png"
# IMAGE_PATH = "src/samples/test.jpg"

img = cv2.imread(IMAGE_PATH)
print(f"Image loaded: {img is not None}")
print(f"Image shape: {img.shape if img is not None else 'N/A'}")

thresh = preprocess_image(img)
print(f"Threshold shape: {thresh.shape}")

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Total contours found: {len(contours)}")

digit_contours = []
for i, c in enumerate(contours):
    x,y,w,h = cv2.boundingRect(c)
    aspect = h / float(w) if w > 0 else 0
    valid = is_valid_digit_contour(c, thresh.shape[0])
    print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}, aspect={aspect:.2f}, h_ratio={h/thresh.shape[0]:.2f}, valid={valid}")
    if valid:
        digit_contours.append(c)

print(f"\nValid digit contours: {len(digit_contours)}")

# Save debug image
debug_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
cv2.drawContours(debug_img, digit_contours, -1, (0, 0, 255), 3)
cv2.imwrite("debug_output.png", debug_img)
print("Debug image saved to debug_output.png")
