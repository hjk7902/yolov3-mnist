import numpy as np

YOLO_STRIDES  = [8, 16, 32]
YOLO_ANCHORS  = [[[10,  13], [16,   30], [33,   23]],
                 [[30,  61], [62,   45], [59,  119]],
                 [[116, 90], [156, 198], [373, 326]]]
STRIDES       = np.array(YOLO_STRIDES)
ANCHORS       = (np.array(YOLO_ANCHORS).T/STRIDES).T

NUM_CLASS     = 10 # COCO 데이터셋이면 80, MNIST 데이터셋이면 10
