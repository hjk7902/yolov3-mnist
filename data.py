import os
import random
import numpy as np
import cv2

from config import *
from image_process import *
from bbox_iou import *

YOLO_STRIDES = [8, 16, 32]
YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],                [[30, 61], [62, 45], [59, 119]],        [[116, 90], [156, 198], [373, 326]]]

STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T / STRIDES).T


# 파일에서 클래스 라벨을 읽어 딕셔너리로 만들어 반환
def read_class_names(class_label_path):
    names = {}
    with open(class_label_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


class DataGenerator(object):
    def __init__(self,
                 data_path,                 annot_path,                 class_label_path,                 load_images_to_ram=True,                 data_aug=True,                 input_size=416,                 anchor_per_scale=3,                 max_bbox_per_scale=100,                 batch_size=4,                 strides=STRIDES,                 anchors=ANCHORS):
        self.input_size = input_size
        self.annot_path = annot_path
        self.batch_size = batch_size
        self.data_aug = False
        self.strides = strides
        self.classes = read_class_names(class_label_path)
        self.num_classes = len(self.classes)
        self.anchors = anchors
        self.anchor_per_scale = anchor_per_scale
        self.max_bbox_per_scale = max_bbox_per_scale
        self.load_images_to_ram = load_images_to_ram
        self.annotations = self.load_annotations(annot_path)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size)) 
        self.batch_count = 0 
        self.output_sizes = input_size // strides

    # 아노테이션 경로에서 데이터파일을 읽어옴 
    def load_annotations(self, annot_path):
        # C:\mnist_test\000009.jpg 
        # 156,153,178,175,9 278,294,300,316,0 
        annotations = []

        with open(self.annot_path, 'r') as f:
            # 파일에서 데이터를 불러와 라인별로 자름 
            data = f.read().splitlines()

        # 공백으로 잘라 맨 앞의 파일경로제외하고  
        # 길이가0이 아닌 행들을 리스트로 만들어 놓음 
        # 파일명만 있는 행 제거 
        # (객체가 없는 이미지의 어노테이션 데이터임) 
        lines = [line.strip() for line in data                 if len(line.strip().split()[1:]) != 0]

        # 랜덤하게 섞음 
        np.random.shuffle(lines)

        for line in lines:
            # 공백으로 나눔 
            # 예: line=['C:\mnist_test\000009.jpg', 
            # 156,153,178,175,9', '278,294,300,316,0'] 
            annotation = line.split()
            image_path = annotation[0]

            # 어노테이션 이미지파일이 없으면 예외 발생시킴 
            if not os.path.exists(image_path):
                raise KeyError(f"{image_path} 파일이 없음")

            # 램 사용하면 이미지를 메모리에 저장 후 사용 
            # 램 사용하지 않으면 
            #    __next__에서 parse_annotation을 실행, 
            #    parse_annotation에서 이미지가 로드됨 
            if self.load_images_to_ram:
                image = cv2.imread(image_path)
            else:
                image = '' 

            # [['C:\mnist_test\000009.jpg', 
            # [156,153,178,175,9', '278,294,300,316,0'], ''], ... ] 
            annotations.append([image_path, annotation[1:],                                image])

        return annotations

    # 아노테이션 데이터 파싱 
    def parse_annotation(self, annotation, mAP='False'):
        if self.load_images_to_ram:
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path) # 이미지를 불러옴 

        #  [[156,153,178,175,9], [278,294,300,316,0]] 
        bboxes = np.array([list(map(int, box.split(',')))                           for box in annotation[1]])

        # 이미지 증강 - 숫자, 문자는 좌/우 반전이 필요 없음 
        # 이미지를 변환하면 경계 상자도 같이 바꿔줘야 함 
        if self.data_aug:
            # 좌/우 반전(생략) 
#             image, bboxes = random_horizontal_flip( 
#                 np.copy(image), np.copy(bboxes)) 
            # 자르기 
            image, bboxes = random_crop(np.copy(image),                                         np.copy(bboxes))  
            # 이동 
            image, bboxes = random_translate(np.copy(image),                                              np.copy(bboxes))

        # mAP=False이면 원본 이미지를 입력 이미지 크기로 변환 
        if not mAP:
            square_shape = [self.input_size, self.input_size]
            image, bboxes = self.ip.resize_to_squre(                 np.copy(image), square_shape, np.copy(bboxes))

        return image, bboxes
 
    # 상자 전처리 
    def preprocess_true_boxes(self, bboxes):
        # 스트라이드의 수 만큼 출력 레벨이 만들어짐 
        OUTPUT_LEVELS = len(self.strides)

        # output_size = 416/[8, 16, 32] = [52, 26, 13] -> N
        # anchor_per_scale = 3, num_classes = 10(MNIST일 경우)
        # 출력 레벨 수 만큼 (N,N,3,15) 모양의 라벨 배열 초기화
        label = [np.zeros((self.output_sizes[i],                           self.output_sizes[i],                           self.anchor_per_scale,                           5 + self.num_classes))                  for i in range(OUTPUT_LEVELS)]
        # max_bbox_per_scale = 100 
        # 출력 레벨 수 만큼 (100,4) 모양 경계상자 배열 초기화 
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4))                       for _ in range(OUTPUT_LEVELS)]
        # 출력 레벨 수 만큼 상자 수 배열 초기화 
        bbox_count = np.zeros((OUTPUT_LEVELS,))

        # 모든 상자 수 만큼 실행 
        for bbox in bboxes:
            # 상자 좌표 
            bbox_coor = bbox[:4]
            # 상자 클래스 라벨 
            bbox_class_ind = bbox[4]
            # 상자의 클래스 라벨 원-핫 인코딩
            onehot = np.zeros(self.num_classes, dtype=np.float64) 
            onehot[bbox_class_ind] = 1.0

            # 원-핫 라벨 평활화(Label Smoothing) 
            # 레이블 정규화라고 부르기도 함 
            # 손실함수가 cross entropy이고,
            # 활성화 함수를 softmax를 사용할 때 적용 
            # 가장 큰 벡터가 나머지 벡터보다 커지는 것을 억제 
            # 공식: y_ls = (1-alpha)*y_onehot + alpha/K 
            K = self.num_classes
            alpha = 0.01 
            smooth_onehot = (1-alpha)*onehot + alpha/K 

            # 상자 좌표를 상자 x,y,w,h로 변환 후 표준화 
            bbox_xywh = np.concatenate(                [(bbox_coor[2:] + bbox_coor[:2]) * 0.5,                   bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(OUTPUT_LEVELS):  # range(3): 
                # 앵커박스 
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(                      bbox_xywh_scaled[i, 0:2]).astype(np.int32)+0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                # 실제 박스와 앵커박스 IoU계산 
                iou_scale = bbox_iou(                    bbox_xywh_scaled[i][np.newaxis, :],                    anchors_xywh)
                iou.append(iou_scale)

                # IoU가 0.3 이상인 박스만 처리함 
                iou_mask = iou_scale > 0.3 
                if np.any(iou_mask):
                    xi, yi = np.floor(                        bbox_xywh_scaled[i, 0:2]).astype(np.int32) 

                    label[i][yi, xi, iou_mask, :] = 0 
                    label[i][yi, xi, iou_mask, 0:4] = bbox_xywh
                    label[i][yi, xi, iou_mask, 4:5] = 1.0 
                    label[i][yi, xi, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(                        bbox_count[i]%self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1 
                    exist_positive = True 
  
            if not exist_positive:
                bst_anc_idx = np.argmax(np.array(iou).reshape(-1),                                           axis=-1)
                best_detect = int(bst_anc_idx / self.anchor_per_scale)
                best_anchor = int(bst_anc_idx % self.anchor_per_scale)
                xi, yi = np.floor(                     bbox_xywh_scaled[best_detect,                                      0:2]).astype(np.int32)

                label[best_detect][yi, xi, best_anchor, :] = 0 
                label[best_detect][yi, xi,                                   best_anchor, 0:4] = bbox_xywh 
                label[best_detect][yi, xi,                                   best_anchor, 4:5] = 1.0 
                label[best_detect][yi, xi,                                   best_anchor, 5:] = smooth_onehot 

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh 
                bbox_count[best_detect] += 1 

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        output_boxes = label_sbbox, label_mbbox, label_lbbox,                        sbboxes, mbboxes, lbboxes
        return output_boxes 

    def __len__(self):
        return self.num_batchs
  
    def __iter__(self):
        return self 
 
    # 배치 크기만큼 이미지와 라벨 박스를 반환 
    def __next__(self):
        with tf.device('/cpu:0'):
            # 배치 이미지를 갖는 배열 
            batch_image = np.zeros(            (self.batch_size,             self.input_size,             self.input_size,             3), dtype=np.float32)

            # 배치 라벨(small, middle, large) 경계 상자 
            batch_label_sbbox = np.zeros(            (self.batch_size,             self.output_sizes[0],              self.output_sizes[0],             self.anchor_per_scale,              5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros(            (self.batch_size,             self.output_sizes[1],              self.output_sizes[1],             self.anchor_per_scale,              5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros(            (self.batch_size,             self.output_sizes[2],              self.output_sizes[2],             self.anchor_per_scale,              5 + self.num_classes), dtype=np.float32)

            # 배치 크기만큼 경계 상자를 저장할 변수 
            batch_sbboxes = np.zeros(            (self.batch_size,              self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros(            (self.batch_size,              self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros(            (self.batch_size,             self.max_bbox_per_scale, 4), dtype=np.float32)

            exceptions = False 
            num = 0 
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:  # 배치 크기만큼 실행 
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: 
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation) 
                    try:
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes) 
                    except IndexError:
                        exceptions = True 
                        print("IndexError,", annotation[0])

                    batch_image[num, :, :, :] = image
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    num += 1

                if exceptions:
                    print('\n')
                    raise Exception("데이터셋에 문제가 있습니다.")

                self.batch_count += 1 
                batch_sm_target = batch_label_sbbox, batch_sbboxes
                batch_md_target = batch_label_mbbox, batch_mbboxes
                batch_lg_target = batch_label_lbbox, batch_lbboxes

                target = (batch_sm_target, batch_md_target, batch_lg_target)
                return batch_image, target
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration