import numpy as np

YOLO_STRIDES  = [8, 16, 32]
YOLO_ANCHORS  = [[[10,  13], [16,   30], [33,   23]],
                 [[30,  61], [62,   45], [59,  119]],
                 [[116, 90], [156, 198], [373, 326]]]
STRIDES       = np.array(YOLO_STRIDES)
ANCHORS       = (np.array(YOLO_ANCHORS).T/STRIDES).T

NUM_CLASS          = 10                # 클래스 라벨의 수
DOMAIN             = "mnist"           # 데이터 도메인 이름, mnist 숫자 분류이면 mnist

WARMUP_EPOCHS      = 2                 # 웜업 이폭
EPOCHS             = 100               # 학습 횟수, 자동생성된 mnist train 데이터가 1000개면 100번정도
LOGDIR             = "logs"            # 학습로그를 저장할 디렉토리
SAVE_BEST_ONLY     = True              # val loss가 가장 좋은 모델을 저장, True 권장
SAVE_CHECKPOINT    = False             # True이면 학습 시 모든 유효한 모델을 저장함, False 권장
CHECKPOINTS_FOLDER = "checkpoints"     # 모델이 저장될 디렉토리
MODEL_NAME         = DOMAIN+"_custom"  # 저장될 모델의 이름
SCORE_THRESHOLD    = 0.3               # IoU 임계값

TRAIN_DATA_PATH    = "data/"+DOMAIN+"/train"
TRAIN_ANNOT_PATH   = "data/"+DOMAIN+"/train.txt"
TEST_DATA_PATH     = "data/"+DOMAIN+"/test"
TEST_ANNOT_PATH    = "data/"+DOMAIN+"/test.txt"
CLASS_LABEL_PATH   = "data/"+DOMAIN+"_names.txt"

