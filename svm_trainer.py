import os.path as osp
import os
import cv2
import numpy as np
from sklearn.svm import SVC
import skimage
import pickle


POSITIVE = "cropped_data/positive"
NEGATIVE = "cropped_data/negative"
NAMES = ["waldo", "wenda", "wizard"]
CLASSES = ["waldo", "wenda", "wizard", "negative"]
CLASS_ID = {
    "waldo": 0,
    "wenda": 1,
    "wizard": 2,
    NEGATIVE: 3,
}


def read_files(dir):
    files = [i for i in os.listdir(dir)]
    images = []
    for file in files:
        image = cv2.imread(osp.join(dir, file))
        images.append(image)
    return images


def to_hog(images):
    hog = cv2.HOGDescriptor()
    gs = [cv2.resize(image, (128, 128)) for image in images]
    images = [hog.compute(image) for image in gs]
    return images


def main():
    # Open test data
    data = {}
    for name in NAMES:
        dir = osp.join(POSITIVE, name)
        data[name] = read_files(dir)

    data[NEGATIVE] = read_files(osp.join(NEGATIVE))

    for key in data:
        data[key] = to_hog(data[key])

    print("hog computed")

    X = []
    y = []
    for classification in data:
        for features in data[classification]:
            X.append(features)
            y.append(CLASS_ID[classification])

    print('data collected')
    print('start training SVM')

    X = np.array(X).reshape((len(X), -1))
    y = np.array(y)

    print(X.shape, y.shape)

    clf = SVC(gamma='auto')
    clf.fit(X, y)
    with open("svc_model.bat", "wb") as f:
        f.write(pickle.dumps(clf))


if __name__ == "__main__":
    main()
