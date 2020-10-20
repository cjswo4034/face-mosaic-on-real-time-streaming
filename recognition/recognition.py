from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
from scipy import misc
import numpy as np

import time
import detection, facenet, utils
import os
import pickle


class Recognition:
    def __init__(self, sess, recognition_threshold=0.85, resize_rate=0.2,
                 model_name="20180402-114759", classifier_name='result1'):
        self.sess = sess

        print('initializing parameters')
        model_dir = 'model/{}/{}.pb'.format(model_name, model_name)
        classifier_filename = 'classifier/classifier_{0}.pkl'.format(classifier_name)
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        self.image_size = 160
        self.resize_rate = resize_rate
        self.thresholdRec = recognition_threshold  # Probability threshold

        print('loading facenet network from {}'.format(model_dir))
        facenet.load_model(model_dir)

        print('loading your embedding vectors')
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

        print('loading classifier from {}'.format(classifier_filename_exp))
        with open(classifier_filename_exp, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)
            self.HumanNames = self.class_names
            self.HumanNames.append('Unknown')

    def number_human(self):
        return self.HumanNames

    # recognize face
    def recognize_faces(self, img, bounding_boxes):
        person_count = [0 for i in range(len(self.HumanNames))]

        predictions = []  # prediction of each faces
        best_class_indices = []  # class index
        best_class_probabilities = []  # best case's probability

        for i, bb in enumerate(bounding_boxes):
            cropped = img[bb[1]: bb[3], bb[0]: bb[2], :]
            cropped = facenet.flip(cropped, False)

            predictions.append(self.recognize_face(cropped))

            best_probability = np.max(predictions[i], axis=0)

            if best_probability < self.thresholdRec:  # 정확도가 기준치 미만이면 unknown
                best_class_indices.append(len(self.HumanNames) - 1)
                best_class_probabilities.append(best_probability)
            else:
                best_class_indices.append(int(np.argmax(predictions[i], axis=0)))
                best_class_probabilities.append(predictions[i][best_class_indices[i]])
                person_count[best_class_indices[i]] += 1

        return self.get_info(len(bounding_boxes), person_count, best_class_indices, best_class_probabilities)

    def recognize_face(self, croped_image) -> '입력된 얼굴의 예측 Class':
        emb_array = np.zeros((1, self.embedding_size))

        scaled = misc.imresize(croped_image, (self.image_size, self.image_size), interp='bilinear')
        scaled = cv2.resize(scaled, (self.image_size, self.image_size),
                            interpolation=cv2.INTER_CUBIC)
        scaled = facenet.prewhiten(scaled)
        scaled_reshape = scaled.reshape(-1, self.image_size, self.image_size, 3)

        feed_dict = {self.images_placeholder: scaled_reshape, self.phase_train_placeholder: False}

        emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)

        return self.model.predict_proba(emb_array)[0]

    # return bounding boxes and tag vectors
    # face_area : detected area
    def get_info(self, bbs_len, person_count, best_class_indices, best_class_probabilities):
        for i in range(bbs_len):
            if person_count[best_class_indices[i]] > 1 and best_class_indices[i] != len(self.HumanNames) - 1:
                for j in range(bbs_len):
                    if best_class_probabilities[i] < best_class_probabilities[j]:
                        person_count[best_class_indices[i]] -= 1
                        person_count[len(self.HumanNames) - 1] += 1
                        best_class_indices[i] = len(self.HumanNames) - 1

        match_names = list(map(lambda x: self.HumanNames[x], best_class_indices))
        return match_names, best_class_probabilities


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
    recognition = Recognition(sess=sess, recognition_threshold=recognition_threshold,
                              resize_rate=resize_rate, model_name=model, classifier_name="test_2")

    bounding_boxes = match_names = p = []

    video = cv2.VideoCapture("../video/test.mp4")

    print("Start Reading...")
    while True:
        _, img = video.read()
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)

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

        cv2.imshow("origin", img)
        cv2.waitKey(0)
