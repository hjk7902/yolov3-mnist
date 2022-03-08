import tensorflow as tf

from image_process import resize_to_square
from data import read_class_names
from post_process import *
from yolov3 import Create_YOLOv3

yolo = Create_YOLOv3(num_class=10)
yolo.load_weights("checkpoints/mnist_custom")
weights = yolo.get_weights()
class_names = read_class_names("mnist.names")

cap = cv2.VideoCapture(1)
if cap.isOpened():
    while True:
        yolo.set_weights(weights)
        ret, image = cap.read()
        if not ret:
            print("프레임을 받지 못했습니다.")
            break 

        # 밝기를 100만큼 더함 
        dummy = np.full(image.shape, fill_value=100, dtype=np.uint8)
        cv2.add(image, dummy, image)
                
        # 콘트라스트 강조함 
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # 이미지를 정사각형 모양으로 만듬 
        image_data = resize_to_square(np.copy(image), 416)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # 상자 예측 
        pred_box = yolo.predict(image_data)
        pred_box = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_box]
        pred_box = tf.concat(pred_box, axis=0)
        
        # 상자 후처리 
        bboxes = postprocess_boxes(pred_box, image, 416, 0.3)

        # NMS에 의해 해당 영역에서 상자 하나만 남김 
        bboxes = nms(bboxes, 0.45, method="nms")

        # 상자를 그림 
        image = draw_bbox(image, bboxes, class_names)

        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
else:
    print('연결된 카메라가 없습니다.')

cap.release()
cv2.destroyAllWindows()