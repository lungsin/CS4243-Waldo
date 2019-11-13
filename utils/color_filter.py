import cv2
import numpy as np

from matplotlib import pyplot as plt


def red_filter(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 150, 40])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 150, 40])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output = np.ones(img.shape)
    output[np.where(mask == 0)] = 0

    return output


def white_filter(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 250], dtype=np.uint8)
    upper_white = np.array([255, 250, 255], dtype=np.uint8)

    mask = cv2.inRange(img_hsv, lower_white, upper_white)

    # set my output img to zero everywhere except my mask
    output = np.ones(img.shape)
    output[np.where(mask == 0)] = 0

    return output


fmap = {
    "red": red_filter,
    "white": white_filter,
}

if __name__ == "__main__":
    import sys
    img = cv2.imread(f"../datasets/JPEGImages/{sys.argv[2]}.jpg")
    img = fmap[sys.argv[1]](img)
    plt.imshow(img)
    plt.show()
