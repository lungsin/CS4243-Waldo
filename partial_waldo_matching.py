import cv2
import numpy as np
from util import resize, display, pyramid
from utils.nms import non_max_suppression_fast

def template_matching(image, template, method = cv2.TM_CCOEFF_NORMED, threshold = 0.5): 
    w, h = template.shape[1], template.shape[0]
    
    print(image.shape)
    res = cv2.matchTemplate(image, template, method)
    print(np.min(res), np.max(res))
    loc = np.where(res >= threshold)
    
    print(len(loc))
    total = []
    for pt in zip(*loc[::-1]):
        total.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
    
    return np.asarray(total)

def partial_matching_waldo_pyramid(image, template, threshold=0.375, should_resize=True):
    template = cv2.imread(f"datasets/Features/{template}.jpg")
    if should_resize:
        template = resize(template, 128)
    if image.shape[0] < 6000:
        image = resize(image, 6000)
    
    total = []
    for i in pyramid(image, downscale=1.2, min_size=template.shape): 
        boxes = template_matching(i, template, threshold=threshold) * (image.shape[0] / i.shape[0])
        total.extend(boxes)
    
    total = np.asarray(total)
    total = non_max_suppression_fast(total, 0)
    return total


def find_waldo(image):
    total = []
    total.extend(partial_matching_waldo_pyramid(image, "waldo_face_eye", 0.5))

    total.extend(partial_matching_waldo_pyramid(image, "027_0", 0.4))
    total.extend(partial_matching_waldo_pyramid(image, "027_0", 0.4, should_resize=False))

    total.extend(partial_matching_waldo_pyramid(image, "waldo_face_eye"))
    total.extend(partial_matching_waldo_pyramid(image, "waldo_face_eye", should_resize=False))

    total.extend(partial_matching_waldo_pyramid(image, "000_0", 0.5))
    total.extend(partial_matching_waldo_pyramid(image, "000_0", 0.5, should_resize=False))

    waldo_body = partial_matching_waldo_pyramid(image, "078_0", 0.3)
    if len(waldo_body) > 0:
        waldo_body[:, 1] -= waldo_body[:, 2] - waldo_body[:, 0]
        waldo_body[:, 3] -= waldo_body[:, 2] - waldo_body[:, 0]
        total.extend(waldo_body)

    waldo_body = partial_matching_waldo_pyramid(image, "078_0", 0.3, should_resize=False)
    if len(waldo_body) > 0:
        waldo_body[:, 1] -= waldo_body[:, 2] - waldo_body[:, 0]
        waldo_body[:, 3] -= waldo_body[:, 2] - waldo_body[:, 0]
        total.extend(waldo_body)

    total = np.asarray(total)
    total = non_max_suppression_fast(total, 0)
    return total


def find_wenda(image):
    total = []
    total.extend(partial_matching_waldo_pyramid(image, "wenda", threshold=0.4))
    total = np.asarray(total)
    total = non_max_suppression_fast(total, 0)
    return total


if __name__ == "__main__":
    import sys

    img = cv2.imread(f"datasets/JPEGImages/{sys.argv[1]}.jpg")

    total = []
    total.extend(partial_matching_waldo_pyramid(img, sys.argv[2], 0.45))
    total = np.asarray(total)
    total = non_max_suppression_fast(total, 0)

    for j in total:
        img2 = cv2.rectangle(img, (j[0], j[1]), (j[2], j[3]), color=(0, 255, 0), thickness=10)
    display(img)