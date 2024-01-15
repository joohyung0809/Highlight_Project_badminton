import cv2
import os
import numpy as np
import pandas as pd
import re

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,  
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

# 각 파일 path
protoFile = "C:/Users/joohy/Computer_Vision/openpose-master/models/pose_deploy_linevec.prototxt"
weightsFile = "C:/Users/joohy/Computer_Vision/openpose-master/models/pose_iter_440000.caffemodel"

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 디렉토리 내의 모든 이미지 파일 가져오기
directory = "C:/Users/joohy/Computer_Vision/yolov4-deepsort/jump/"
image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

# csv_file_path = "C:/Users/joohy/Computer_Vision/yolov4-deepsort/jump/captured_frames.csv"
# df = pd.read_csv(csv_file_path, header=None)


# 이미지 파일에 대한 작업 수행
for image_file in image_files:
    image_path = os.path.join(directory, image_file)

    # 이미지 읽어오기
    image = cv2.imread(image_path)

    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    imageHeight, imageWidth, _ = image.shape

    # network에 넣기위해 전처리
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

    # network에 넣어주기
    net.setInput(inpBlob)

    # 결과 받아오기
    output = net.forward()

    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
    H = output.shape[2]
    W = output.shape[3]

    # 키포인트 검출시 이미지에 그려줌
    points = []
    for i in range(0, 15):
        probMap = output[0, i, :, :]

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H

        if prob > 0.1:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Head와 RWrist의 좌표 가져오기
    neck_coord = points[BODY_PARTS["Neck"]]
    rwrist_coord = points[BODY_PARTS["RWrist"]]
    lwrist_coord = points[BODY_PARTS["LWrist"]]
    RElbow_coord = points[BODY_PARTS["RElbow"]]
    LElbow_coord = points[BODY_PARTS["LElbow"]]

    RHip_coord = points[BODY_PARTS["RHip"]]
    Chest_coord = points[BODY_PARTS["Chest"]]
    RShoulder_coord = points[BODY_PARTS["RShoulder"]]
    
    
    BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,  
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}

    # Head와 RWrist가 모두 검출되고, RWrist가 Head보다 위에 있으면 이미지 삭제
    #          2                 3                4               7                 4                  2                            4                    2                            4                  7                   7                     6                           7                1
    if RShoulder_coord and RElbow_coord and rwrist_coord and lwrist_coord and (rwrist_coord[1] < RShoulder_coord[1]) and (abs(rwrist_coord[0] - RShoulder_coord[0]) < 70) and (rwrist_coord[1] <= lwrist_coord[1] and (abs(lwrist_coord[0] - LElbow_coord[0]) < 25) and (lwrist_coord[0] > neck_coord[0])):
    # if neck_coord and rwrist_coord and neck_coord[1] > rwrist_coord[1] and lwrist_coord and neck_coord[1] > lwrist_coord[1]:
        pass
    else:
        # image_number = re.search(r'\d+', image_file).group()
        
        # print(image_number)
        # # 이미지 번호에 해당하는 행 삭제
        # df = df[df[df.columns[0]] == int(image_number)]

        # # 업데이트된 DataFrame을 CSV 파일에 저장
        # df.to_csv(csv_file_path, index=False, header=None)
        
        os.remove(image_path)
        print(f"Deleted: {image_file}")

# 작업 완료 후 메시지 출력
print("작업 완료")
