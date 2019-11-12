import os.path as osp
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import util
from matplotlib import pyplot as plt

def create_dataset(anno_file, img_file, id):
    print('creating dataset for image ', id)
    anno_tree = ET.parse(anno_file)
    objs = anno_tree.findall('object')

    size = anno_tree.find('size')
    size_x, size_y = int(size.find('width').text), int(size.find('height').text)


    box = {}
    for idx, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        name = obj.find('name').text
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)

        if name in box:
            box[name].append((x1, y1, x2, y2))
        else:
            box[name] = [(x1, y1, x2, y2)]

    image = np.asarray(cv2.imread(img_file))

    box_flatten = []
    for name in box:
        import os
        if not os.path.exists('cropped_data/positive/' + name):
            os.mkdir('cropped_data/positive/' + name)

        for i, rect in enumerate(box[name]):
            x1, y1, x2, y2 = rect
            box_flatten.append(rect)
            out_filename = osp.join('cropped_data', 'positive', name, "{}_{}.jpg".format(id, i))
            print(out_filename)
            subimage = image[y1:y2, x1:x2]
            cv2.imwrite(out_filename, subimage)

    import os
    if not os.path.exists('cropped_data/negative/'):
        os.mkdir('cropped_data/negative/')

    selected = []
    for img in util.pyramid(image):
        for x, y, window in util.slidingWindow(img, 50, (100, 100)):
            # if (cnt > 100): break
            x2, y2 = x + 100, y + 100

            if np.var(window) < 3500: continue
            window_box = (x, y, x2, y2)

            iou_max = 0.
            for anno in box_flatten:
                anno_box = np.array(anno)
                anno_box = np.divide(anno_box, img.shape[0] / image.shape[0])

                iou_max = max(iou_max, util.bb_intersection_over_union(window_box, anno_box))

            if iou_max < 0.2:
                selected.append(window)

    print(selected)
    selected = np.random.choice(selected, 250)
    cnt = 0
    for i in selected:
        out_filename = osp.join('cropped_data', 'negative', "{}_{}.jpg".format(id, cnt))
        cnt += 1

        print(out_filename)
        print(np.var(i))
        cv2.imwrite(out_filename, i)


def process(image_id):
    """
    :param image_id:
    :return:
    """
    anno_dir = 'datasets/Annotations'
    anno_file = osp.join(anno_dir, '{}.xml'.format(image_id))

    img_dir = 'datasets/JPEGImages'
    img_file = osp.join(img_dir, '{}.jpg'.format(image_id))
    assert osp.exists(anno_file), '{} not find.'.format(anno_file)
    assert osp.exists(img_file), '{} not find.'.format(img_file)

    create_dataset(anno_file, img_file, image_id)

def main():
    # Generate test data
    with open("datasets/ImageSets/train.txt", 'r') as f :
        data = f.readlines()
        for i in range(len(data)): data[i] = data[i].strip()

        for id in data:
            print("image id: {}".format(id))
            process(id)


if __name__=='__main__':
    main()