from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2

import time
from recognition import detection, facenet, utils, recognition
import os


if __name__ == "__main__":
    print(os.getcwd())
    model = '20180402-114759'

    print("Create Session")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    recognition_threshold = 0.85
    conf_threshold = 0.7
    resize_rate = 0.5

    print("Load Network")
    detection = detection.Detection(sess=sess, resize_rate=resize_rate, conf_threshold=conf_threshold)
    recognition = recognition.Recognition(sess=sess, recognition_threshold=recognition_threshold,
                              resize_rate=resize_rate, model_name=model, classifier_name="test_2")

    bounding_boxes = match_names = p = []

    video = cv2.VideoCapture("../video/sample4.mp4")

    print("Start Reading...")
    while True:
        _, img = video.read()

        height, width, channel = img.shape
        # matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 270, 1)
        # frame = cv2.warpAffine(frame, matrix, (width, height))

        tic = time.time()
        resize_img = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

        if resize_img.ndim == 2:
            resize_img = facenet.to_rgb(resize_img)
        resize_img = resize_img[:, :, 0:3]

        bounding_boxes = detection.detect_faces(resize_img, img.shape)

        if bounding_boxes.shape[0] > 0:
            match_names, p = recognition.recognize_faces(img, bounding_boxes)
        else:
            bounding_boxes = match_names = p = []
        toc = time.time() - tic

        img = utils.mosaic(img, bounding_boxes, match_names, 6)
        img = utils.draw_box(img, bounding_boxes, match_names, p)

        cv2.imshow("test", img)
        cv2.waitKey(1)