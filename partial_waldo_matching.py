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
        
def partial_matching_waldo_pyramid(image, template):
    template = cv2.imread(f"datasets/Features/{template}.jpg")
    template = resize(template, 128)
    if image.shape[0] < 6000:
        image = resize(image, 6000)
    
    total = []
    for i in pyramid(image, downscale=1.2, min_size=template.shape): 
        boxes = template_matching(i, template, threshold=0.375) * (image.shape[0] / i.shape[0])
        total.extend(boxes)
    
    total = np.asarray(total)
    total = non_max_suppression_fast(total, 0)
    return total
    
if __name__ == "__main__":
    import sys

    img = cv2.imread(f"datasets/JPEGImages/{sys.argv[1]}.jpg")
    # img = cv2.imread(f"cropped_data/positive/waldo/{sys.argv[1]}.jpg")

    total = []
    total.extend(partial_matching_waldo_pyramid(img, "027_0"))
    total.extend(partial_matching_waldo_pyramid(img, "waldo_face_eye"))
    total = np.asarray(total)
    total = non_max_suppression_fast(total, 0)

    for j in total:
        img2 = cv2.rectangle(img, (j[0], j[1]), (j[2], j[3]), color=(0, 255, 0), thickness=10)
    display(img)
