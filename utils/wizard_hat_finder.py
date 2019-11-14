import cv2
import numpy as np
from utils.color_filter import blue_filter
from matplotlib import pyplot as plt
from utils.pattern_finder import color_with_pyramid


def wizard_hat_filter(image):
    template = cv2.imread(f"datasets/Features/wizard_hat.jpg")
    template = cv2.resize(template, (60, 60))
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    kernel = np.zeros(template.shape)
    kernel[template > 245] = 1

    return color_with_pyramid(image, [blue_filter], kernel,
                              is_absolute=False,
                              threshold=700,
                              debug=True,
                              box_size=template.shape[0]
                              )


if __name__ == "__main__":
    import sys

    img = cv2.imread(f"../datasets/JPEGImages/{sys.argv[1]}.jpg")

    total = wizard_hat_filter(img)
    for j in total:
        img2 = cv2.rectangle(img, (j[1], j[0]), (j[3], j[2]), color=(0, 255, 0), thickness=10)
    plt.imshow(img)
    plt.show()
