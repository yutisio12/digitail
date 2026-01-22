import cv2
from utils import match_digit

def read_digit(digit_img):
    digit_img = cv2.resize(digit_img, (100, 200))
    h, w = digit_img.shape

    segments_roi = [
        ((0.25*w, 0.03*h), (0.75*w, 0.18*h)),   # top
        ((0.78*w, 0.18*h), (0.95*w, 0.48*h)),   # top-right
        ((0.78*w, 0.52*h), (0.95*w, 0.85*h)),   # bottom-right
        ((0.25*w, 0.85*h), (0.75*w, 0.97*h)),   # bottom
        ((0.05*w, 0.52*h), (0.22*w, 0.85*h)),   # bottom-left
        ((0.05*w, 0.18*h), (0.22*w, 0.48*h)),   # top-left
        ((0.25*w, 0.45*h), (0.75*w, 0.60*h)),   # middle
    ]

    segments = []

    for (x1,y1),(x2,y2) in segments_roi:
        roi = digit_img[int(y1):int(y2), int(x1):int(x2)]
        ratio = cv2.countNonZero(roi) / roi.size

        if ratio > 0.18:
            segments.append(1)
        elif ratio > 0.07:
            segments.append(0.5)
        else:
            segments.append(0)

    # VALIDASI DIGIT 1 (LCD)
    right_on = segments[1] >= 0.5 and segments[2] >= 0.5
    left_on = segments[5] >= 0.5 or segments[4] >= 0.5

    if right_on and not left_on:
        return "1", 0.95

    return match_digit(segments)
