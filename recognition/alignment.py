from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import cv2
import os
import shutil
import tensorflow as tf
import numpy as np
from recognition import detect_face, facenet

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QSizePolicy, QPushButton, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage


class Alignment:
    def __init__(self, sess, pnet, rnet, onet,
                 out_path='output', data_path='dataset'):
        self.sess = sess
        self.pnet, self.rnet, self.onet = pnet, rnet, onet

        self.output_path = os.path.expanduser(out_path)  # 정렬된 얼굴 이미지가 저장될 경로
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.dataset = facenet.get_dataset(data_path)    # 인식 대상이 저장된 이미지 경로
        self.minsize = 20                                # 사진에서 얼굴 영역의 최소 크기 (20 x 20)
        self.threshold = [0.6, 0.7, 0.7]                 # 얼굴 탐지 시 각 단계별 경계값
        self.factor = 0.709                              # 얼굴 탐지에 필요한 변수
        self.margin = 44                                 # 얼굴 탐지에 필요한 변수
        self.image_size = 182                            # 저장될 얼굴 사진의 크기

        self.images = []           # 자장된 얼굴 사진들의 경로들이 저장될 배열
        self.aligned_images = 0    # 최종적으로 변환된 얼굴 이미지 개수

    def remove_data(self):         # 변환된 사진들을 제거하기 위한 함수
        for cls in self.dataset:
            output_class_path = os.path.join(self.output_path, cls.name)
            if os.path.exists(output_class_path) and os.path.isdir(output_class_path):
                try:
                    shutil.rmtree(output_class_path)
                except Exception as e:
                    print(e)
                    continue
            print('removed {} from "{}"'.format(cls.name, self.output_path))

    def aline_face(self, img, bounding_boxes):  # 입력된 이미지(img)와 해당 이미지에서 찾은 얼굴들 중 하나의 얼굴만 추출
        det = bounding_boxes[:, 0:4]            # bounding_boxes에서 x1, y1, x2, y2만 이용
        img_size = np.asarray(img.shape)[0:2]
        if bounding_boxes.shape[0] > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])  # 얼굴 영역의 크기
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)                # 이미지 중심과 얼굴 중심의 거리
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)       # 크기 및 거리를 고려해서 얼굴 중 하나를 선택
            det = det[index, :]
        det = np.squeeze(det)
        bb_temp = np.zeros(4, dtype=np.int32)

        bb_temp[0] = det[0]
        bb_temp[1] = det[1]
        bb_temp[2] = det[2]
        bb_temp[3] = det[3]

        return img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]

    def make_align(self):      # 이미지에서 존재하는 얼굴 중 가장 기준에 부합하는 얼굴사진만 저장하기 위한 함수
        total_images = 0

        for cls in self.dataset:    # 인식 대상이 저장된 경로에 존재하는 폴더명
            output_class_path = os.path.join(self.output_path, cls.name)

            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)

            for image_path in cls.image_paths:      # 데이터셋에 존재하는 폴더내의 사진들
                total_images += 1
                output_filename = os.path.join(output_class_path, cls.name)  # 저장될 경로 및 파일 이름

                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        error_message = '{}: {}'.format(image_path, e)
                        print(error_message)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)  # 흑백 이미지
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                            print('to_rgb data dimension: ', img.ndim)
                        img = img[:, :, 0:3]

                        bounding_boxes, _ = detect_face.detect_face(img,
                                                                    self.minsize, self.pnet, self.rnet, self.onet,
                                                                    self.threshold, self.factor)
                        if bounding_boxes.shape[0] > 0:         # 사진에서 발견된 얼굴 영역의 개수가 1개 이상일 때
                            cropped_temp = self.aline_face(img, bounding_boxes)     # 이미지에 존재하는 얼굴 중 하나를 추출
                            try:
                                scaled_temp = misc.imresize(cropped_temp, (self.image_size, self.image_size),
                                                            interp='bilinear')      # 얼굴 이미지를 182x182로 변경
                                print(image_path)
                                cv2.imshow('awef', scaled_temp)
                                cv2.waitKey(0)
                                scaled_flip = np.fliplr(scaled_temp)                # 얼굴 좌우 반전 (데이터셋 증폭용)
                                temp_name = '{}_{}.{}'.format(output_filename, self.aligned_images, 'png')
                                flip_name = '{}_{}.{}'.format(output_filename, self.aligned_images + 1, 'png')
                                self.images.append(temp_name)   # 잘못된 얼굴 사진을 구별하기 위해 경로를 저장
                                self.images.append(flip_name)
                                self.aligned_images += 2

                                misc.imsave(temp_name, scaled_temp)
                                misc.imsave(flip_name, scaled_flip)
                            except Exception as e:
                                print(e)
                                continue
                        else:
                            print('Unable to align {}'.format(image_path))

        print('Total number of images: {}'.format(total_images))
        print('Number of successfully aligned images: {}'.format(self.aligned_images))

    def get_image_paths(self):      # 저장된 얼굴 사진들의 경로와 갯수를 반환
        return self.images, self.aligned_images


class App(QWidget):
    def __init__(self, align):
        QWidget.__init__(self)

        self.image_paths, self.total_images = align.get_image_paths()
        self.check_img = [True for i in range(len(self.image_paths))]   # 이미지의 제거 유무, True : 존재함, False : 제거됨
        self.idx = 0   # 현재 확인중인 이미지의 index
        self.first_idx, self.last_idx = 0, self.total_images - 2        # 이미지들 중 가장 처음 및 마지막 이미지 Index
        self.count = self.total_images      # 제거되지 않은 이미지들의 갯수

        self.msgbox = QMessageBox()
        self.imageLabel = QLabel()          # 현재 확인중인 이미지를 표시하는 Label
        self.cnt_label = QLabel()           # 현재 확인중인 이미지 Index 및 전체 이미지를 표시하는 Label
        self.init_ui()

    def show_messagebox(self, messagge):
        msg = QMessageBox(self)
        msg.setWindowTitle('Image viewer')
        msg.setText(messagge)
        msg.move(QApplication.primaryScreen().size().width() / 2, QApplication.primaryScreen().size().height() / 2)
        msg.exec_()

    def delete(self):
        f1 = self.image_paths[self.idx]         # 이미지를 좌우반전 시켰으므로 원본 및 번전된 이미지 경로를 가져옴
        f2 = self.image_paths[self.idx + 1]
        if os.path.exists(f1):                  # 이미지가 존재하면
            os.remove(f1)                       # 제거한다
            self.check_img[self.idx] = False    # 이미지 삭제 표시
            self.count -= 1                     # 제거되지 않은 이미지 갯수 조정
        if os.path.exists(f2):
            os.remove(f2)
            self.check_img[self.idx + 1] = False
            self.count -= 1
        if self.idx == self.first_idx:          # 제거된 이미지가 남아있는 이미지 중 첫번째 이미지라면
            while self.count > 0 and not self.check_img[self.first_idx]:    # 남아있는 이미지 중 가장 첫번째 이미지의 Index를
                self.first_idx += 2             # first_idx 변수에 할당한다
        if self.idx == self.last_idx:           # 제거된 이미지가 남아있는 이미지 중 마지막 이미지라면
            while self.count > 0 and not self.check_img[self.last_idx]:     # 남아있는 이미지 중 가장 마지막 이미지의 Index를
                self.last_idx -= 2              # last_idx 변수에 할당한다
        if self.count < 1:                      # 모든 이미지를 제거한 경우 종료
            self.show_messagebox('존재하는 이미지가 없습니다.')
            QWidget.close(self)
            return
        self.next()                             # 이미지를 제거한 뒤 다음 이미지 노출하기 위함

    def prev(self):             # 이전 이미지를 화면에 출력하기 위한 함수
        self.idx -= 2           # 이전 이미지의 경로를 가져오기 위함
        if self.idx >= 0 and not self.check_img[self.idx]:          # 현재 확인중인 이미지의 바로 이전 이미지가 존재하지 않을 경우
            while self.idx >= 0 and not self.check_img[self.idx]:   # 이전 이미지 중 가장 가깝고 제거되지 않은 이미지의 Index
                self.idx -= 2
        if self.idx < self.first_idx:               # 남아있는 이미지 중에서 첫번째 이미지보다 더 적은 이미지를 확인하려는 경우
            self.show_messagebox('First image')     # 경고를 표시함
            self.idx = self.first_idx               # 맨 처음 이미지 Index를 대입
        current_img_path = self.image_paths[self.idx]       # 이미지의 경로 가져오기
        self.load(current_img_path)                         # 해당 이미지를 화면에 표시
        self.cnt_label.setText('{} / {}'.format(int(self.idx / 2) + 1, int(self.total_images / 2)))
        # 변경된 이미지의 Index를 Label에 표시

    def next(self):             # 다음 이미지를 화면에 출력하기 위한 함수
        self.idx += 2           # 다음 이미지의 경로를 가져오기 위함
        if self.idx < self.total_images and not self.check_img[self.idx]:         # 현재 확인중인 이미지의 바로 다음 이미지가 존재하지 않을 경우
            while self.idx < self.total_images and not self.check_img[self.idx]:  # 다음 이미지 중 가장 가깝고 제거되지 않은 이미지의 Index
                self.idx += 2
        if self.idx > self.last_idx:               # 남아있는 이미지 중에서 마지막 이미지의 다음 이미지를 확인하려는 경우
            if self.exit('마지막 이미지입니다. 종료하시겠습니까?'):  # 종료 여부
                QWidget.close(self)
                return
            else:
                self.idx = self.last_idx                    # 마지막 이미지 Index를 대입
        current_img_path = self.image_paths[self.idx]       # 이미지의 경로 가져오기
        self.load(current_img_path)                         # 해당 이미지를 화면에 표시
        self.cnt_label.setText('{} / {}'.format(int(self.idx / 2) + 1, int(self.total_images / 2)))
        # 변경된 이미지의 Index를 Label에 표시

    def load(self, fileName):      # 입력받은 이미지를 화면에 출력하기 위한 함수
        image = QImage(fileName)
        if image.isNull():         # 파일이 존재하지 않을 경우 경고 표시 후 빈화면 출력
            QMessageBox.information(self, QApplication.applicationName(),
                                    "Cannot load " + fileName)
            self.setPixmap(QPixmap())

        self.imageLabel.setPixmap(QPixmap.fromImage(image))

    def exit(self, msg):        # 종료창이 떴을 때 사용자가 누른 버튼을 확인하기 위한 함수
        self.msgbox.setText(msg)
        return self.msgbox.exec_() == QMessageBox.Yes

    def init_ui(self):
        self.setWindowTitle('Image viewer')

        self.msgbox.setWindowTitle('Image viewer')  # 종료창
        self.msgbox.setText( '종료하시겠습니까?')
        self.msgbox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        self.imageLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)     # 이미지가 그려지는 공간
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setPixmap(QPixmap.fromImage(QImage(self.image_paths[0])))

        delete_button = QPushButton("Delete image")     # 이미지 삭제 버튼
        prev_button = QPushButton(" <<< ")              # 이전 이미지 버튼
        next_button = QPushButton(" >>> ")              # 다음 이미지 버튼
        self.cnt_label.setText('{} / {}'.format(self.idx + 1, int(self.total_images / 2)))
        # 현재 출력중인 이미지 Index 및 전체 이미지 Index

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(prev_button)
        btn_layout.addStretch()                 # 버튼을 가운데 정렬하기 위한 빈 공간
        btn_layout.addWidget(delete_button)
        btn_layout.addWidget(self.cnt_label)
        btn_layout.addStretch()
        btn_layout.addWidget(next_button)

        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        delete_button.clicked.connect(self.delete)
        prev_button.clicked.connect(self.prev)
        next_button.clicked.connect(self.next)
        self.resize(300, 330)
        self.move(QApplication.primaryScreen().size().width() / 2 - 150,
                  QApplication.primaryScreen().size().height() / 2 - 165)


if __name__ == "__main__":
    print('Creating networks and loading parameters')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    pnet, rnet, onet = detect_face.create_mtcnn(sess, 'model')
    align = Alignment(sess, pnet, rnet, onet)
    align.make_align()

    app = QApplication(sys.argv)
    mainWindow = App(align)
    mainWindow.show_messagebox("인식 대상의 얼굴이 아닌 사진을 지워주세요!!")
    mainWindow.show()
    app.exec_()