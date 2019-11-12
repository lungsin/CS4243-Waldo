import os.path as osp
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import numpy as np
import cv2
from argparse import ArgumentParser
import math


def draw_bbox(pimage, captions):

    h,w,c = pimage.shape
    src_image = pimage.copy()
    image = pimage.copy()

    pad = 1
    for b in range(len(captions)):
       font = 0.7
       caption = captions[b]
       object_name = caption['name']
       bbox = [int(b) for b in caption['bbox']]
       score = str('%.4f'% caption['score'])
       color= [255,255,255]
       cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, -1)
       bbox_caption = object_name+':'+score
       text_len = len(bbox_caption)*10
       text_color = [0]*3
       if (bbox[1] > 20):
          if bbox[0] + text_len <= w:
              cv2.rectangle(image, (bbox[0]-pad, bbox[1]-18),(bbox[0]+text_len, bbox[1]), color, -1)
              cv2.putText(image, bbox_caption, (bbox[0],bbox[1]-6), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          font, text_color)
          else:
              cv2.rectangle(image, (bbox[2] - text_len-pad, bbox[1] - 18), (bbox[2], bbox[1]), color, -1)
              cv2.putText(image, bbox_caption, (bbox[2]-text_len, bbox[1] - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          font, text_color)
       else:
          if bbox[0] + text_len <= w:
              cv2.rectangle(image, (bbox[0]-pad, bbox[1]),(bbox[0]+text_len, bbox[1]+20), color, -1)
              cv2.putText(image, bbox_caption, (bbox[0],bbox[1]+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, font,
                          text_color)
          else:
              cv2.rectangle(image, (bbox[2] - text_len-pad, bbox[1]), (bbox[2], bbox[1] + 20), color, -1)
              cv2.putText(image, bbox_caption, (bbox[2]-text_len, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          font,text_color)

    cv2.addWeighted(image, 0.8, src_image, 0.2, 0, src_image)

    return src_image

def cascade_face_detection(image_file, anno_file):
    """
    Visualize annotations
    :param image_file:
    :param anno_file:
    :return:
    """

    image = np.asarray(cv2.imread(image_file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # face_cascade = cv2.CascadeClassifier('pretrained/haarcascades/haarcascade_profileface.xml')
    # face_cascade = cv2.CascadeClassifier('pretrained/haarcascades/haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier('pretrained/haarcascades/haarcascade_eye.xml')
    # face_cascade = cv2.CascadeClassifier('pretrained/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    # face_cascade = cv2.CascadeClassifier('pretrained/lbpcascades/lbpcascade_frontalface.xml')

    face_cascade = cv2.CascadeClassifier('pretrained/lbpcascades/lbpcascade_animeface.xml')

    # face_cascade = cv2.CascadeClassifier('pretrained/lbpcascades/lbpcascade_frontalface_improved.xml')

    faces = face_cascade.detectMultiScale(gray,
                                            scaleFactor = 1.01,
                                            minNeighbors = 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # anno_tree = ET.parse(anno_file)
    # objs = anno_tree.findall('object')
    # anno = []
    # for idx, obj in enumerate(objs):
    #     name = obj.find('name').text
    #     bbox = obj.find('bndbox')
    #     x1 = float(bbox.find('xmin').text)
    #     y1 = float(bbox.find('ymin').text)
    #     x2 = float(bbox.find('xmax').text)
    #     y2 = float(bbox.find('ymax').text)
    #     anno.append({'name': name, 'score': 1, 'bbox': [x1, y1, x2, y2]})
    #
    # image = draw_bbox(image, anno)

    # image = cv2.resize(image, (1280, 900))
    cv2.imshow('face detection', image)
    cv2.waitKey()

def image_edge(image):
    image = cv2.GaussianBlur(image, (13, 13), 2)
    image = cv2.Canny(image, 50, 200)
    # image = cv2.Laplacian(image, cv2.CV_64F)
    return image

def template_matching(image_file, anno_file):
    """
    Visualize annotations
    :param image_file:
    :param anno_file:
    :return:
    """

    image = np.asarray(cv2.imread(image_file))
    image = cv2.resize(image, (1920, 1280))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('pretrained/template/waldo_big.jpg',0)
    # template = image_edge(template)
    # gray = image_edge(gray)

    # cv2.imshow("canny template", template)
    # cv2.imshow("canny gray image", gray)
    # cv2.waitKey()
    w, h = template.shape[::-1]
    print(f"image shape: {gray.shape}")
    print(f"template shape: {w, h}")
    ori_w, ori_h = w, h

    scale = 1.2
    for i in range(20):
        resized = cv2.resize(template, (w, h))

        if  gray.shape[0] < w or gray.shape[1] < h:
            w = math.floor(w / scale)
            h = math.floor(h / scale)
            continue

        res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)

        # visualize max loc
        # clone = np.dstack([gray, gray, gray])
        # cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
        #               (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)
        # cv2.imshow(f"Visualize, val: {maxVal}", clone)
        # cv2.waitKey(0)

        threshold = 0.5
        # finding the values where it exceeds the threshold
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            # draw rectangle on places where it exceeds threshold
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

        w = math.floor(w / scale)
        h = math.floor(h / scale)

    cv2.imshow('template matching', image)
    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    cv2.waitKey()

def main(image_id):
    """
    :param image_id:
    :return:
    """
    image_dir = 'datasets/JPEGImages'
    anno_dir = 'datasets/Annotations'
    image_file = osp.join(image_dir,'{}.jpg'.format(image_id))
    anno_file = osp.join(anno_dir, '{}.xml'.format(image_id))
    assert osp.exists(image_file),'{} not find.'.format(image_file)
    assert osp.exists(anno_file), '{} not find.'.format(anno_file)
    # cascade_face_detection(image_file, anno_file)
    template_matching(image_file, anno_file)

if __name__ == "__main__":
    parser = ArgumentParser(description='visualize annotation for image.')
    parser.add_argument('-imageID', dest='imageID', default='079',help='input imageID, e.g., 001')
    args = parser.parse_args()
    main(args.imageID)