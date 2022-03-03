import os
import shutil
import tensorflow as tf
from data import DataGenerator
from config import *
from bbox_iou import bbox_iou, bbox_giou

NUM_CLASS = 10

LOGDIR = "logs" # 학습로그를 저장할 디렉토리

WARMUP_EPOCHS = 2
EPOCHS = 100

SAVE_BEST_ONLY        = True              # val loss가 가장 좋은 모델을 저장, True 권장
SAVE_CHECKPOINT       = False             # True이면 학습 시 모든 유효한 모델을 저장함, False 권장
CHECKPOINTS_FOLDER    = "checkpoints"     # 모델이 저장될 디렉토리
MODEL_NAME            = "mnist_custom"    # 저장될 모델의 이름
SCORE_THRESHOLD        = 0.3


if os.path.exists(LOGDIR): shutil.rmtree(LOGDIR)
writer = tf.summary.create_file_writer(LOGDIR)

validate_writer = tf.summary.create_file_writer(LOGDIR)

from yolov3 import Create_YOLOv3
yolo = Create_YOLOv3(train_mode=True, num_class=NUM_CLASS)


trainset = DataGenerator(data_path="/mnist_train", annot_path="mnist_train.txt", class_label_path="mnist.names")
testset = DataGenerator(data_path="/mnist_test", annot_path="mnist_test.txt", class_label_path="mnist.names")
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = WARMUP_EPOCHS * steps_per_epoch
total_steps = EPOCHS * steps_per_epoch

optimizer = tf.keras.optimizers.Adam()
    
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs {gpus}')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass


def compute_loss(pred, conv, label, bboxes, i=0, num_class=80, iou_loss_thresh=0.45):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_class))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    # bbox_iou
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # 실제 상자에서 가장 큰 예측값을 갖는 상자로 IoU 값 찾기
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # 가장 큰 iou가 임계값보다 작으면 예측 상자에 개체가 포함되지 않은 것으로 간주되고 배경 상자로 설정
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < iou_loss_thresh, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Confidence의 loss 계산
    # 그리드에 객체가 포함된 경우 1, 그렇지 않을경우 0
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss


def train_step(image_data, target, num_class=80, lr_init=1e-4, lr_end=1e-6):
    with tf.GradientTape() as tape:
        pred_result = yolo(image_data, training=True)
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        grid = 3
        for i in range(grid):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i, num_class=NUM_CLASS)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, yolo.trainable_variables)
        optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

        # 학습률 갱신
        # about warmup: https://arxiv.org/pdf/1812.01187.pdf
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * lr_init
        else:
            lr = lr_end + 0.5 * (lr_init - lr_end)*(
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
        optimizer.lr.assign(lr.numpy())

        # Loss를 log에 저장
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()
        
    return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()
	
	
def validate_step(image_data, target, num_class=80):
    with tf.GradientTape() as tape:
        pred_result = yolo(image_data, training=False)
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        grid = 3 
        for i in range(grid):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i, num_class=num_class)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        
    return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()


def main() :

    best_val_loss = 1000 # should be large at start
    save_directory = os.path.join(CHECKPOINTS_FOLDER, MODEL_NAME)

    for epoch in range(EPOCHS):
        for image_data, target in trainset:
            results = train_step(image_data, target, num_class=NUM_CLASS)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))

        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(CHECKPOINTS_FOLDER, MODEL_NAME))
            continue
        
        count = 0
        giou_val, conf_val, prob_val, total_val = 0, 0, 0, 0
        
        for image_data, target in testset:
            results = validate_step(image_data, target, num_class=NUM_CLASS)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
            
        # validation loss 저장
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()
            
        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val/count, conf_val/count, prob_val/count, total_val/count))

        if SAVE_CHECKPOINT and not SAVE_BEST_ONLY:
            save_directory = os.path.join(CHECKPOINTS_FOLDER, MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count))
            yolo.save_weights(save_directory)
        if SAVE_BEST_ONLY :
            if(best_val_loss>total_val/count):
                yolo.save_weights(save_directory)
                best_val_loss = total_val/count
                

if __name__=="__main__":
    main()