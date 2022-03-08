import numpy as np

YOLO_STRIDES  = [8, 16, 32]
YOLO_ANCHORS  = [[[10,  13], [16,   30], [33,   23]],
                 [[30,  61], [62,   45], [59,  119]],
                 [[116, 90], [156, 198], [373, 326]]]
STRIDES       = np.array(YOLO_STRIDES)
ANCHORS       = (np.array(YOLO_ANCHORS).T/STRIDES).T

NUM_CLASS          = 10
WARMUP_EPOCHS      = 2
EPOCHS             = 100
LOGDIR             = "logs" # 학습로그를 저장할 디렉토리
SAVE_BEST_ONLY     = True              # val loss가 가장 좋은 모델을 저장, True 권장
SAVE_CHECKPOINT    = False             # True이면 학습 시 모든 유효한 모델을 저장함, False 권장
CHECKPOINTS_FOLDER = "checkpoints"     # 모델이 저장될 디렉토리
MODEL_NAME         = "mnist_custom"    # 저장될 모델의 이름
SCORE_THRESHOLD    = 0.3