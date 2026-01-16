# make_dataset.py
import cv2, numpy as np, random

for i in range(1000):
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    liter = f"{random.randint(0,999)}.{random.randint(0,99):02}"
    price = str(random.randint(1000,99999))

    cv2.putText(img, liter, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
    cv2.putText(img, price, (100,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)

    cv2.imwrite(f"data/images/{i}.jpg", img)
