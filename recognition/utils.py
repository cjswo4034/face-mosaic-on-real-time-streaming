import cv2
import time


def probablities(target, previos):  # 현재 박스 좌표와 이전 박스의 일치 정도
    if target[0] > previos[2] or target[2] < previos[0]\
            or target[1] > previos[3] or target[3] < previos[1]:  # 두 개의 박스가 겹치지 않는다
        return False

    tar_area = (target[2] - target[0]) * (target[3] - target[1])
    pre_area = (previos[2] - previos[0]) * (previos[3] - previos[1])
    if tar_area / pre_area > 1.3 or tar_area / pre_area < 0.7:   # 두 개의 박스 넓이 차이가 크다
        return False

    x1 = max(target[0], previos[0])
    y1 = max(target[1], previos[1])
    x2 = min(target[2], previos[2])
    y2 = min(target[3], previos[3])

    area = ((x2 - x1) * (y2 - y1)) / ((target[2] - target[0]) * (target[3] - target[1]))    # 겹치는 비율
    if area > 0.6:
        return True
    else:
        return False


def resize_rate(shape):         # 전송된 이미지의 사이즈별로 축소 비율 조정
    if min(shape) >= 1080:      # FHD
        resize_rate = 0.25
    elif min(shape) >= 720:     # HD
        resize_rate = 0.4
    else:
        resize_rate = 0.5
    return resize_rate


def draw_box(img, bb, match_names, p):    # img에 인식 영역 및 결과 및 확률 그리기
    color = (0,0,255)
    draw_text(img, '{0:.3f}'.format(time.perf_counter() * 1000))
    for idx, box in enumerate(bb):
        x = bb[idx][0] + 10
        y = bb[idx][1] + 10
        nx = bb[idx][2] + 10
        ny = bb[idx][3]
        cv2.rectangle(img, (x, y), (nx, ny), color, 2)
        if match_names[idx] == 'Unknown' or match_names[idx] == 'temp':
            text = 'Unknown'
        else:
            text = '{0} : {1}%'.format(match_names[idx], int(p[idx] * 100))
        draw_text(img, text, (x, ny + 30))
    return img


def draw_text(img, text, position=None, color=(0, 0, 255)):
    if not position:
        position = (img.shape[0] + 30, 30)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_COMPLEX,
                1, color, thickness=2, lineType=2)
    return img


def mosaic(img, bb, best_class_indices, mosaic_rate):     # Frame에서 지정된 영역을 모자이크 처리
    for idx in range(len(bb)):
        if best_class_indices[idx] == 'Unknown' or best_class_indices[idx] == 'temp':
            [x, y, nx, ny] = bb[idx]
            w = nx - x
            h = ny - y
            face_img = img[y:ny, x:nx]

            # 영역 축소
            image = cv2.resize(face_img, (mosaic_rate, mosaic_rate))            # 해당 영역을 축소한 후
            # 영역 보간법을 이용하여 원본 크기로 확대
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)     # 원본 크기로 확대
            img[y:ny, x:nx] = image

    return img


def expand_size(bounding_boxies, expand_rate):      # 축소된 Bounding box를 원본 Frame에 맞추기 위한 매핑
    for idx in range(len(bounding_boxies)):
        bounding_boxies[idx][0] = int(bounding_boxies[idx][0] * expand_rate)
        bounding_boxies[idx][1] = int(bounding_boxies[idx][1] * expand_rate)
        bounding_boxies[idx][2] = int(bounding_boxies[idx][2] * expand_rate)
        bounding_boxies[idx][3] = int(bounding_boxies[idx][3] * expand_rate)


if __name__ == "__main__":
    img = cv2.imread('dataset/kim/1.jpg')

    target = [204, 37, 257, 125]
    previos = [224, 38, 269, 127]

    x1 = max(target[0], previos[0])
    y1 = max(target[1], previos[1])
    x2 = min(target[2], previos[2])
    y2 = min(target[3], previos[3])

    cv2.rectangle(img, (target[0], target[1]), (target[2], target[3]), (0, 0, 255), 2)
    cv2.rectangle(img, (previos[0], previos[1]), (previos[2], previos[3]), (255, 255, 255), 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), -1)

    area = ((x2 - x1) * (y2 - y1)) / ((target[2] - target[0]) * (target[3] - target[1]))
    print(area)
    area = ((x2 - x1) * (y2 - y1)) / ((previos[2] - previos[0]) * (previos[3] - previos[1]))
    print(area)

    cv2.imshow('test', img)
    cv2.waitKey(0)
