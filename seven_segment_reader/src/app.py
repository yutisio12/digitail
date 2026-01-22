import cv2
from preprocess import preprocess_image
from segment_reader import read_digit
from utils import is_valid_digit_contour

# IMAGE_PATH = "src/samples/gambar1.png"   # 8
IMAGE_PATH = "src/samples/test.jpg"   # 153

def main(image_path):
    img = cv2.imread(image_path)
    thresh = preprocess_image(img)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    digit_contours = []
    for c in contours:
        if is_valid_digit_contour(c, thresh.shape[0]):
            digit_contours.append(c)

    # sort kiri → kanan
    digit_contours = sorted(
        digit_contours,
        key=lambda c: cv2.boundingRect(c)[0]
    )

    digits = []
    confidences = []

    for cnt in digit_contours:
        x,y,w,h = cv2.boundingRect(cnt)

        # perlakuan khusus digit 1 (sangat ramping)
        if w < thresh.shape[1] * 0.03:
            digits.append("1")
            confidences.append(0.98)
            continue

        pad = int(h * 0.1)
        y1 = max(0, y - pad)
        y2 = min(thresh.shape[0], y + h + pad)

        digit_img = thresh[y1:y2, x:x+w]
        digit_img = cv2.resize(digit_img, (100, 200))

        d, conf = read_digit(digit_img)
        digits.append(d)
        confidences.append(conf)

    print("====================")
    print("HASIL BACA :", "".join(digits))
    print("CONFIDENCE :", confidences)
    print("====================")

if __name__ == "__main__":
    main(IMAGE_PATH)
