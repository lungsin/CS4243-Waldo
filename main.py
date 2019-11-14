import os
import sys

import cv2

from partial_waldo_matching import find_waldo, find_wenda
from utils.wizard_hat_finder import wizard_hat_filter
from matplotlib import pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage python ./main.py [image_file]")
        return

    file_name = sys.argv[1]
    assert os.path.exists(file_name)
    print(f"Finding Waldo and friends in {file_name}")

    image = cv2.imread(file_name)

    # waldo
    waldo = find_waldo(image)
    for i in waldo:
        cv2.putText(image, "Waldo", (i[0], i[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2,
                    lineType=cv2.LINE_AA)
        cv2.rectangle(image, (i[0], i[1]), (i[2], i[3] + 2*(i[2] - i[0])), (0, 255, 0), thickness=10)

    # wenda
    wenda = find_wenda(image)
    for i in wenda:
        cv2.putText(image, "Wenda", (i[0], i[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2,
                    lineType=cv2.LINE_AA)
        cv2.rectangle(image, (i[0], i[1]), (i[2], i[3] + 2*(i[2] - i[0])), (0, 255, 0), thickness=10)

    # wizard
    wizard = wizard_hat_filter(image)
    for i in wizard:
        cv2.putText(image, "Wizard", (i[1], i[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2,
                    lineType=cv2.LINE_AA)
        cv2.rectangle(image, (i[1], i[0]), (i[3], i[2] + 2*(i[3] - i[1])), (0, 255, 0), thickness=10)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()