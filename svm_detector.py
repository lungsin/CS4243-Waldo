import pickle
import cv2
import os.path as osp
import numpy as np
import util

from matplotlib import pyplot as plt

model = None


def detect(anno_file, img_file, id):
    global model
    image = np.asarray(cv2.imread(img_file))

    plt.imshow(image)
    plt.show()

    for img in util.pyramid(image):
        raw = [i for i in util.slidingWindow(img, 50, (100, 100))]
        hog_window = [cv2.resize(window, (128, 128)) for _, _, window in raw]
        hog_window = [cv2.HOGDescriptor().compute(i).reshape((-1)) for i in hog_window]
        detection = util.splitter(hog_window, model.predict)
        print(detection)
        with_id = zip(range(0, len(raw)), detection)
        with_id = [i for i in with_id if i[1] != 3]
        for res in with_id:
            print(res)


def process(image_id, model):
    anno_dir = 'datasets/Annotations'
    anno_file = osp.join(anno_dir, '{}.xml'.format(image_id))

    # img_dir = 'datasets/JPEGImages'
    img_dir = 'cropped_data/positive/waldo'
    img_file = osp.join(img_dir, '{}.jpg'.format(image_id))
    # assert osp.exists(anno_file), '{} not find.'.format(anno_file)
    assert osp.exists(img_file), '{} not find.'.format(img_file)

    detect(anno_file, img_file, image_id)


def main():
    global model
    with open("svc_model.bat", "rb") as f:
        model = pickle.load(f)

    print(model)

    with open("datasets/ImageSets/train.txt", 'r') as f:
        data = f.readlines()
        for i in range(len(data)): data[i] = data[i].strip()

        for id in data:
            print("image id: {}".format(id))
            process(id + "_0", model)
            break


if __name__ == "__main__":
    main()
