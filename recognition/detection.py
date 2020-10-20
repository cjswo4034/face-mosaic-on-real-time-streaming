import tensorflow as tf
import cv2
import time
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Detection:        # 입력받은 이미지에서 얼굴 영역을 반환하기 위한 Class
    def __init__(self, sess, model_path='model', resize_rate=0.25, conf_threshold=0.7):
        self.sess = sess

        # CV2 DNN
        self.model_path = '{}/opencv_face_detector_uint8.pb'.format(model_path)  # Network Weight
        self.config_path = '{}/opencv_face_detector.pbtxt'.format(model_path)  # Network Configuration File
        self.net = cv2.dnn.readNetFromTensorflow(self.model_path, self.config_path)  # Load Network

        self.conf_threshold = conf_threshold    # 발견된 영역들 중에서 얼굴 영역이라고 판단되는 정확도의 하한
        self.resize_rate = resize_rate      # 입력받은 이미지 축소 비율

    def detect_faces(self, img, main_img_size):
        h, w, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), [104, 117, 123], False, False)  # Network 입력으로 사용하기 위해 변환
        self.net.setInput(blob)

        detections = self.net.forward()         # 미리 훈련된 네트워크 모델로 사진 입력 후 영역 추출
        boudning_boxes = [bb[3:7] for bb in detections[0, 0] if bb[2] > self.conf_threshold]
        # 영역 표현 형식을 [Index][x1, y1, x2, y2] 로 변경하기 위한 로직
        # cv2_dnn 출력 형식 : [?][?][Index][?, ?, confidence, x1, y1, x2, y2] (각 좌표는 좌측 상단 및 우측 하단 좌표를 비율로 표현)
        det = []

        expand_rate = 1 / self.resize_rate
        for idx, bb in enumerate(boudning_boxes):
            if bb[0] > 1 or bb[1] > 1 or bb[2] > 1 or bb[3] > 1: continue
            x1 = int(max(bb[0] * main_img_size[1] + 0.5, 0))
            y1 = int(max(bb[1] * main_img_size[0] + 0.5, 0))
            x2 = int(min(bb[2] * main_img_size[1] + 0.5, main_img_size[1]))
            y2 = int(min(bb[3] * main_img_size[0] + 0.5, main_img_size[0]))

            # 박스의 가로 길이가 일정 이상일 때 scale up 할 시 변하는 위치를 조정하기 위한 로직
            #if (x2 - x1) >= 80 * expand_rate: (x1, x2) = (int(x1 + 16 * expand_rate), int(x2 + 16 * expand_rate))

            det.append([x1, y1, x2, y2])

        return np.array(det)  # 얼굴 영역들에 대한 결과값

    def draw_box(self, img, bounding_boxes):
        for bb in bounding_boxes:
            x = int(bb[0])
            y = int(bb[1])
            nx = int(bb[2])
            ny = int(bb[3])
            cv2.rectangle(img, (x, y), (nx, ny), (0, 0, 255), 2)
        return img

    def put_text(self, img, text, position, mode):
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), thickness=2, lineType=2)


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    resize_rate = 0.5
    detection = Detection(sess, resize_rate=resize_rate, conf_threshold=0.7)

    video = cv2.VideoCapture('../video/{}.{}'.format('test', 'mp4'))

    process_this_img = True
    tot_rate = 0
    tot_cnt = 0
    while True:
        ref, img = video.read()
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)

        tic = time.time()
        if process_this_img:
            resize_img = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

            resize_img = resize_img[:, :, 0:3]

            cv2_tic = time.time()
            boxes_cv2 = detection.detect_faces(resize_img, img.shape)
            cv2_toc = time.time() - cv2_tic

        img = detection.draw_box(img, boxes_cv2)
        text_cv2 = 'processing time : {0:.5f}'.format(cv2_toc)
        detection.put_text(img, text_cv2, (20, 35), 'cv2')

        toc = time.time() - tic
        tot_cnt += 1
        tot_rate += toc

        # sys.stdout.write('\r{0:0.5f}'.format(tot_rate / tot_cnt))

        cv2.imshow('video1', img)
        ch = cv2.waitKey(0)

        if ch == ord('q'):
            break
        elif ch == ord('s'):
            cv2.waitKey(0)

        process_this_img = not process_this_img