import random
import numpy as np
import cv2
import tensorflow as tf

from image_process import resize_to_square
from data import read_class_names
from post_process import *
    
from yolov3 import Create_YOLOv3
    
NUM_CLASS = 10 

def detect_image(model, image_path, output_path, class_label_path, input_size=416, show=False,
                 score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    original_image = cv2.imread(image_path)
    class_names = read_class_names(class_label_path)

    image_data = resize_to_square(np.copy(original_image), target_size=input_size)
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    image = draw_bbox(original_image, bboxes, class_names, rectangle_colors=rectangle_colors)

    if output_path != '': 
        cv2.imwrite(output_path, image)
    if show:
        cv2.imshow("predicted image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


if __name__=="__main__":
    yolo = Create_YOLOv3(num_class=NUM_CLASS)
    yolo.load_weights("checkpoints/mnist_custom")
    weight = yolo.get_weights()


    yolo.set_weights(weight)
    result_image = detect_image(model=yolo, 
                                image_path="mnist_test_c.jpg", 
                                output_path="mnist_test_out.jpg", 
                                class_label_path="mnist.names", 
                                input_size=416, show=True)