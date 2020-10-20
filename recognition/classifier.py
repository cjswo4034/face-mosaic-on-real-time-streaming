from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from recognition import facenet
import os
import math
import pickle
from sklearn.svm import SVC


class Classifier:           # 각 얼굴들을 대표하는 특징들을 구하여 분류모델 생성하기 위한 Class
    def __init__(self, sess,
                 data_path='output', model='20180402-114759',
                 classifier_name='result1'):
        self.sess = sess
        self.dataset = facenet.get_dataset(data_path)       # 얼굴영역만 저장된 폴더들의 경로를 가져온다.
        self.class_names = [cls.name.replace('_', ' ') for cls in self.dataset]     # Dataset 폴더 내 각각의 폴더(Class)명
        self.paths, self.labels = facenet.get_image_paths_and_labels(self.dataset)  # 이미지들의 경로 및 Class 명
        print('Number of classes: %d' % len(self.dataset))
        print('Number of images: %d' % len(self.paths))

        print('Loading feature extraction model')
        model_path = 'model/{}/{}.pb'.format(model, model)      # 네트워크 모델(얼굴인식)이 저장된 경로
        facenet.load_model(model_path)
        self.model = SVC(kernel='linear', probability=True)     # 분류모델 생성
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

        classifier_path = 'classifier/classifier_{0}.pkl'.format(classifier_name)   # 분류모델이 저장될 이름 및 경로
        self.classifier_path = os.path.expanduser(classifier_path)
        self.batch_size = 1000
        self.image_size = 160

    def make_classifier(self):
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        # 이미지들의 총 갯수
        images = len(self.paths)
        # 한 번에 1000개의 이미지를 네트워크에 입력할 때 반복해야될 횟수
        epoch = int(math.ceil(1.0 * images / self.batch_size))
        # 결과값이 저장될 변수 (이미지, 그 이미지의 특징벡터)
        emb_array = np.zeros((images, self.embedding_size))

        for i in range(epoch):
            # 훈련에 사용할 이미지의 시작 index
            start_index = i * self.batch_size
            # 훈련에 사용할 이미지의 마지막 index
            end_index = min((i + 1) * self.batch_size, images)
            # 훈련에 사용할 이미지들의 경로
            paths_batch = self.paths[start_index:end_index]
            # Facenet에 로드
            images = facenet.load_data(paths_batch, False, False, self.image_size)
            # Facenet에 로드
            feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
            # Facenet을 실행하여 계산
            emb_array[start_index:end_index, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)

        self.model.fit(emb_array, self.labels)  # 결과 vector들로 SVC 모델 실행
        self.save_model()                       # 네트워크 가중치 값 저장

    def save_model(self):
        with open(self.classifier_path, 'wb') as out_path:
            pickle.dump((self.model, self.class_names), out_path)
        print('Saved classifier model to {}'.format(self.classifier_path))


if __name__ == '__main__':
    classifier = Classifier(tf.Session(), classifier_name="test_3")
    classifier.make_classifier()