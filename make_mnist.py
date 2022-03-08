import os
import cv2
import numpy as np
import shutil
import random

SIZE = 416 # 학습용 이미지의 크기
TRAIN_IMAGES_NUM = 1000 # 생성할 학습용 이미지의 수
TEST_IMAGES_NUM = 200   # 생성할 평가용 이미지의 수

# 만들어지는 이미지 크기별 최대 개수(small, medium, big)
IMAGE_SIZE_COUNT = [3, 6, 3] 

# 크기(small, medium, big)별 배율
IMAGE_SIZE_RATIOS = [[0.5, 0.8], [1., 1.5, 2.], [3., 4.]]

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = 255 - x_train
x_test = 255 - x_test

def compute_iou(box1, box2):
    A1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    A2 = (box2[2] - box2[0])*(box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    return  ((xmax-xmin) * (ymax - ymin)) / (A1 + A2)
    
    
def make_image(data, image, label, ratio=1):
    output = data[0]
    boxes = data[1]
    labels = data[2]
    if(len(image.shape)==2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = cv2.resize(image, (int(28*ratio), int(28*ratio)))
    h, w, c = image.shape

    while True:
        xmin = np.random.randint(0, SIZE-w, 1)[0]
        ymin = np.random.randint(0, SIZE-h, 1)[0]
        xmax = xmin + w
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]
        if max(iou) < 0.02:
            boxes.append(box)
            labels.append(label)
            break

    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            output[y][x] = image[j][i]

    return output
    
    
def main():
    for file in ['train','test']:
        images_path = os.getcwd()+f"/mnist_{file}"
        labels_txt = os.getcwd()+f"/mnist_{file}.txt"
        
        if file == 'train': 
            images_num = TRAIN_IMAGES_NUM
            images = x_train
            img_labels = y_train
        if file == 'test': 
            images_num = TEST_IMAGES_NUM
            images = x_test
            img_labels = y_test
            
        if os.path.exists(images_path): shutil.rmtree(images_path)
        os.mkdir(images_path)

        with open(labels_txt, "w") as f:
            image_num = 0
            while image_num < images_num:
                image_path = os.path.realpath(os.path.join(
                    images_path, "%06d.jpg"%(image_num+1)))
                #print(image_path)
                annotation = image_path
                outputs = np.ones(shape=[SIZE, SIZE, 3]) * 255
                bboxes = [[0,0,1,1]]
                labels = [0]
                data = [outputs, bboxes, labels]
                bboxes_num = 0
                
                for i in range(len(IMAGE_SIZE_RATIOS)):
                    N = random.randint(0, IMAGE_SIZE_COUNT[i])
                    if N!=0: bboxes_num += 1
                    for _ in range(N):
                        ratio = random.choice(IMAGE_SIZE_RATIOS[i])
                        idx = random.randint(0, len(images)-1)
                        data[0] = make_image(data, images[idx], img_labels[idx], ratio)

                if bboxes_num == 0: continue
                cv2.imwrite(image_path, data[0]) # 한글경로에 저장 안됨
    #             cv2.imshow("image", data[0])
    #             cv2.waitKey()
    #             cv2.destroyAllWindows()
    #             break
    #             print(image_path, data[0].shape)
                for i in range(len(labels)):
                    if i == 0: continue
                    xmin = str(bboxes[i][0])
                    ymin = str(bboxes[i][1])
                    xmax = str(bboxes[i][2])
                    ymax = str(bboxes[i][3])
                    class_ind = str(labels[i])
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
                image_num += 1
                print('.', end='')
                # print("=> %s" %annotation)
                f.write(annotation + "\n")


if __name__=="__main__":
    main()