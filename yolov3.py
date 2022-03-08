import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

YOLO_STRIDES  = [8, 16, 32]
YOLO_ANCHORS  = [[[10,  13], [16,   30], [33,   23]],
                 [[30,  61], [62,   45], [59,  119]],
                 [[116, 90], [156, 198], [373, 326]]]
STRIDES       = np.array(YOLO_STRIDES)
ANCHORS       = (np.array(YOLO_ANCHORS).T/STRIDES).T


class BatchNormalization(layers.BatchNormalization):
    # "동결 상태(Frozen state)"와 "추론 모드(Inference mode)"는 별개의 개념입니다.
    # 'layer.trainable=False' 이면 레이어를 동결시킵니다. 이것은 훈련하는 동안 내부 상태 즉, 가중치가 바뀌지 않습니다.
    # 그런데 layer.trainable=False이면 추론 모드로 실행됩니다.
    # 레이어는 추론모드에서 현재 배치의 평균 및 분산을 사용하는 대신 현재 배치를 정규화하기 위해 이동 평균과 이동 분산을 사용합니다.
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convolutional(input_layer, filters, kernel_size,
                  downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    kernel_init = tf.random_normal_initializer(stddev=0.01)
    conv = layers.Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides, padding=padding,
                         use_bias=not bn,
                         kernel_initializer=kernel_init,
                         kernel_regularizer=l2(0.0005)
                        )(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate:
        conv = layers.LeakyReLU(alpha=0.1)(conv)

    return conv


def residual_block(input_layer, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters=filter_num1, kernel_size=(1,1))
    conv = convolutional(conv       , filters=filter_num2, kernel_size=(3,3))
    residual_output = short_cut + conv
    return residual_output


def darknet53(input_data):
    input_data = convolutional(input_data, 32, (3,3))
    input_data = convolutional(input_data, 64, (3,3), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  32, 64)

    input_data = convolutional(input_data, 128, (3,3), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 64, 128)

    input_data = convolutional(input_data, 256, (3,3), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, 512, (3,3), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, 1024, (3,3), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 512, 1024)

    return route_1, route_2, input_data


def upsample(input_layer):
    width, height = input_layer.shape[1], input_layer.shape[2]
    output_layer = tf.image.resize(input_layer, (width*2, height*2),
                                   method='nearest')
    return output_layer


def YOLOv3(input_layer, num_class):
    # Darknet-53을 실행하고 그 결과를 받음
    route_1, route_2, conv = darknet53(input_layer)

    conv = convolutional(conv, 512, (1, 1))
    conv = convolutional(conv, 1024, (3, 3))
    conv = convolutional(conv, 512, (1, 1))
    conv = convolutional(conv, 1024, (3, 3))
    conv = convolutional(conv, 512, (1, 1))
    conv_lobj_branch = convolutional(conv, 1024, (3, 3))

    # conv_lbbox는 큰 객체를 예측하기 위해 사용, Shape = [None, 13, 13, 255]
    conv_lbbox = convolutional(conv_lobj_branch,
                               3 * (num_class + 5), (1, 1),
                               activate=False, bn=False)

    conv = convolutional(conv, 256, (1, 1))
    # 최근방법(nearest)을 이용하여 업샘플링
    # 이렇게 하면 업샘플링시 학습이 필요 없으므로 인공신경망 파라미터를 줄인다.
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)
    conv = convolutional(conv, 256, (1, 1))
    conv = convolutional(conv, 512, (3, 3))
    conv = convolutional(conv, 256, (1, 1))
    conv = convolutional(conv, 512, (3, 3))
    conv = convolutional(conv, 256, (1, 1))
    conv_mobj_branch = convolutional(conv, 512, (3, 3))

    # conv_mbbox는 중간 크기 객체를 예측하기 위해 사용, shape = [None, 26, 26, 255]
    conv_mbbox = convolutional(conv_mobj_branch,
                               3 * (num_class + 5), (1, 1),
                               activate=False, bn=False)

    conv = convolutional(conv, 128, (1, 1))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)
    conv = convolutional(conv, 128, (1, 1))
    conv = convolutional(conv, 256, (3, 3))
    conv = convolutional(conv, 128, (1, 1))
    conv = convolutional(conv, 256, (3, 3))
    conv = convolutional(conv, 128, (1, 1))
    conv_sobj_branch = convolutional(conv, 256, (3, 3))

    # conv_sbbox는 작은 객체를 예측하기 위해 사용, shape = [None, 52, 52, 255]
    conv_sbbox = convolutional(conv_sobj_branch,
                               3 * (num_class + 5), (1, 1),
                               activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_output, num_class, i=0):
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size,
                              3, num_class+5))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # 상자의 x, y위치
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # 상자의 가로, 세로 크기
    conv_raw_conf = conv_output[:, :, :, :, 4:5] # 상자의 신뢰도(confidence)
    conv_raw_prob = conv_output[:, :, :, :, 5: ] # 클래스별 확률

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size, dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :],
                      [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # 상자의 중심점을 계산
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # 상자의 너비와 높이를 계산
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf) # 상자의 신뢰도 계산
    pred_prob = tf.sigmoid(conv_raw_prob) # 클래스별 확률 계산

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def Create_YOLOv3(num_class, input_shape=(416,416,3), train_mode=False):
    input_layer  = layers.Input(input_shape)
    conv_tensors = YOLOv3(input_layer, num_class)
    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, num_class, i)
        if train_mode: 
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    model = tf.keras.Model(input_layer, output_tensors)
    return model


