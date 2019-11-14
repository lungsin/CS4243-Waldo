import cv2
import numpy as np
from utils.color_filter import red_filter, white_filter
from matplotlib import pyplot as plt
from utils.pattern_finder import color_with_pyramid


def stripes_filter(image):
    kernel = np.array([0, 0, 0, 0,
                       1, 1, 1, 1,
                       0, 0, 0, 0,
                       1, 1, 1, 1,
                       ]).transpose()
    return color_with_pyramid(image, [red_filter, white_filter], kernel,
                              is_absolute=True,
                              threshold=12,
                              debug=True,
                              box_size=32
                              )


if __name__ == "__main__":
    import sys

    img = cv2.imread(f"../datasets/JPEGImages/{sys.argv[1]}.jpg")

    total = stripes_filter(img)
    for j in total:
        img2 = cv2.rectangle(img, (j[1], j[0]), (j[3], j[2]), color=(0, 255, 0), thickness=10)
    plt.imshow(img)
    plt.show()