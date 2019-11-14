import cv2
import numpy as np

from matplotlib import pyplot as plt

def pyramid(image, downscale=1.5, min_size=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / downscale)
        h = int(image.shape[0] / downscale)
        # image = cv2.resize(image, width=)
        image = cv2.resize(image, dsize=(w, h))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield image


def slidingWindow(image, step, window_size):
    ''' Sliding Window over image - iterable function

    Params:
        image > Image to be processed
        step > Sliding Window step
        windows_size> Sliding Window size
    '''

    # Iterate windows
    for y in range(0, image.shape[0] - window_size[0], step):
        for x in range(0, image.shape[1] - window_size[1], step):
            # Return current window
            yield (x, y, image[y:y + window_size[0], x:x + window_size[1]])


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def splitter(arr, f, size=1000, debug=False):
    res = []
    while len(arr) > size:
        tmp = arr[:size]
        res.extend(f(tmp))
        arr = arr[size:]
        if debug:
            print(tmp, arr, res)

    res.extend(f(arr))
    return res

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
    
def display(img):
    if img.shape[1] > 2560:
        img = resize(img, width=2560)
    if len(img.shape) == 2:
        pass
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if np.max(img) != 0:
        img = img / np.max(img)
    plt.imshow(img)
    plt.show()
        
    
def getPositiveData(name):
    import os
    import matplotlib.pyplot as plt
    path = os.path.join("cropped_data", "positive", name)
    for file in os.listdir(path): 
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        yield img
        
    
def red_filter(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(img_hsv, (0, 150, 40), (10, 255, 255))
    mask1 = cv2.inRange(img_hsv, (170, 150, 40), (180, 255, 255))
    return cv2.bitwise_or(mask0, mask1)


def white_filter(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, (0, 0, 200), (180, 40, 255))
    return mask

def generateEyeFilter(size=(11, 11), r1 = None, r2 = None): 
    if r1 == None and r2 == None: 
        r1 = (size[0] + 1) // 2
        r2 = r1 // 2 + 1
    
    if r1 == None: 
        r1 = r2 * 2 - 1
    
    if r2 == None: 
        r2 = r1 // 2 + 1
        
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)
        
    eyeFilter = np.zeros(size, dtype=np.uint8)
    center = np.array([size[0] // 2, size[1] // 2])
    
    for i in range(size[0]):
        for j in range(size[1]):
            pt = np.array([i, j])
            dis = dist(pt, center)
            if dis <= r1 and dis >= r2:
                eyeFilter[i, j] = 255
    return eyeFilter
