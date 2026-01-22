import cv2

DIGIT_LOOKUP = {
    (1,1,1,1,1,1,0): '0',
    (0,1,1,0,0,0,0): '1',
    (1,1,0,1,1,0,1): '2',
    (1,1,1,1,0,0,1): '3',
    (0,1,1,0,0,1,1): '4',
    (1,0,1,1,0,1,1): '5',
    (1,0,1,1,1,1,1): '6',
    (1,1,1,0,0,0,0): '7',
    (1,1,1,1,1,1,1): '8',
    (1,1,1,1,0,1,1): '9',
}

def match_digit(segments):
    # konversi 0.5 → 1 (LCD tipis)
    binary = tuple(1 if s >= 0.5 else 0 for s in segments)

    if binary in DIGIT_LOOKUP:
        conf = sum(segments) / 7.0
        return DIGIT_LOOKUP[binary], round(conf, 3)

    return '?', 0.0


def is_valid_digit_contour(cnt, img_h):
    x,y,w,h = cv2.boundingRect(cnt)
    aspect = h / float(w)

    if h < img_h * 0.35:
        return False

    if aspect < 1.3:
        return False

    return True
