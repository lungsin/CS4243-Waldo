import cv2
import util
from util import red_filter, white_filter, display, generateEyeFilter
from util import pyramid, slidingWindow
# from utils.color_filter import red_filter, white_filter
import matplotlib.pyplot as plt
import numpy as np

# detect waldo and wenda hat
def detectEyes(window):
    output = window.copy()
    # display(window)
    
    norm = cv2.normalize(window, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    display(norm)
    
    whiteMask = white_filter(norm)
    print("max white mask: {}".format(np.max(whiteMask)))
    display(whiteMask)
    
    eyeFilter = generateEyeFilter(size = (51, 51), r1 = 24, r2 = 12)
    
    display(eyeFilter)
    for x, y, img in slidingWindow(image=whiteMask, step=10, window_size=eyeFilter.shape):
        # print(img.shape, img.dtype)
        # print(eyeFilter.shape, eyeFilter.dtype)
        
        matches = cv2.bitwise_xor(img, eyeFilter)
        
        value = np.sum(matches) / (matches.shape[0] * matches.shape[1] * 255)
        
        # print(value)
        if value > 0.6:
            print(value)
            # print(np.min(img), np.max(img))
            # print(np.min(matches), np.max(matches))
            # print(np.sum(matches), (matches.shape[0] * matches.shape[1]))
            output = whiteMask.copy()
            cv2.rectangle(output, (x, y), (x + eyeFilter.shape[0], y + eyeFilter.shape[1]), (255, 0, 0), 2)
            display(output)
        
        
    display(output)

def main(): 
    for window in util.getPositiveData("waldo"):
        detectEyes(window)

if __name__ == "__main__":
    main()
