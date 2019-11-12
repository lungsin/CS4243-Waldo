import os.path as osp
import xml.etree.ElementTree as ET


def absr(a):
    if a < 0:
        a = -a
    return int(a)


def compute_diff(anno_file):
    anno_tree = ET.parse(anno_file)
    objs = anno_tree.findall('object')
    dx = set([])
    dy = set([])

    size = anno_tree.find('size')

    size_x, size_y = int(size.find('width').text), int(size.find('height').text)
    for idx, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        if absr(x1 - x2) == 14:
            print(anno_file)

        dx.add(size_x // absr(x1 - x2))
        dy.add(size_y // absr(y2 - y1))

    return dx, dy


def compute(image_id):
    """
    :param image_id:
    :return:
    """
    anno_dir = 'datasets/Annotations'
    anno_file = osp.join(anno_dir, '{}.xml'.format(image_id))
    assert osp.exists(anno_file), '{} not find.'.format(anno_file)
    return compute_diff(anno_file)


def main():
    DX, DY = set([]), set([])
    with open("datasets/imageSets/train.txt", 'r') as f:
        data = f.readlines()
        for i in range(len(data)): data[i] = data[i].strip()
        print(data)
        for id in data:
            print("image id: {}".format(id))
            dx, dy = compute(id)
            for x, y in zip(dx, dy):
                DX.add(x)
                DY.add(y)
    DX = sorted(DX)
    DY = sorted(DY)
    print(DX)
    print(DY)


if __name__ == '__main__':
    main()
