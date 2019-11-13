import cv2
import numpy as np
from scipy import ndimage
from util import pyramid
from utils.nms import non_max_suppression_fast


# kernel => -1 means dont care, i (i >= 0) means i-th color
def color_pattern(image, filters, kernel, is_absolute=False, threshold=None):
    total = None
    for i, f in enumerate(filters):
        filtered_image = f(image)

        # generate kernel
        cur_kernel = kernel.copy()
        cur_kernel[kernel == i] = 1
        cur_kernel[kernel != i] = -1
        cur_kernel[kernel == -1] = 0

        filter_result = cv2.filter2D(filtered_image, -1, cur_kernel)
        if total is None:
            total = filter_result
        else:
            total += filter_result

    if is_absolute:
        total = np.absolute(total)

    total_max = ndimage.maximum_filter(total, size=16, mode="constant")

    total[total != total_max] = 0

    if threshold is not None:
        total[total < 8.0] = 0

    res = []
    for i in range(total.shape[0]):
        for j in range(total.shape[1]):
            if total[i][j] > 0:
                res.append([i - 16, j - 16, i + 16, j + 16])

    return np.asarray(res)


def color_with_pyramid(image, filters, kernel, is_absolute=False, threshold=None):
    total = []
    for i in pyramid(image):
        ratio = image.shape[0] / i.shape[0]
        res = color_pattern(i, filters, kernel, is_absolute, threshold) * ratio
        total.extend(res)

    total = np.asarray(total)
    total = non_max_suppression_fast(total, 0)
    return total
