# 이 코드 목적은 B 케이스 해결을 위해 만듦 -> 1,2 / 3 / 4 케이스 구별할 수 있음
# 지금 object-tracker 다음에 실행했음
# 솔직히 내가 얘 실행했는지 안 했는지 모르겠음

import cv2
import pandas as pd
import numpy as np
import os

# 옷 색깔 check
def is_person_wearing_black_color(image_path, target_color, color_range):
    # 이미지를 불러오기
    image = image_path
    print(image.shape)

    # 이미지에서 특정 색상 범위에 해당하는 픽셀 찾기
    lower_color = np.array([max(0, c - color_range) for c in target_color])
    upper_color = np.array([min(255, c + color_range) for c in target_color])
    
    # 이미지에서 색상 범위에 해당하는 픽셀을 찾음
    color_mask = cv2.inRange(image, lower_color, upper_color)

    # 해당 색상을 입은 픽셀의 개수를 세기
    color_pixel_count = np.sum(color_mask > 0)

    # 픽셀 개수를 기준으로 해당 색상의 옷을 입었는지 여부 판단
    threshold_pixel_count = 1000  # 적절한 임계값 설정
    return color_pixel_count #> threshold_pixel_count

def is_person_wearing_color(image_path, target_color, color_range):
    # 이미지를 불러오기
    image = image_path
    # print(image.shape)

    # 이미지에서 특정 색상 범위에 해당하는 픽셀 찾기
    lower_color = np.array([max(0, c - color_range) for c in target_color])
    upper_color = np.array([min(255, c + color_range) for c in target_color])
    
    # 이미지에서 색상 범위에 해당하는 픽셀을 찾음
    color_mask = cv2.inRange(image, lower_color, upper_color)

    # 해당 색상을 입은 픽셀의 개수를 세기
    color_pixel_count = np.sum(color_mask > 0)

    # 픽셀 개수를 기준으로 해당 색상의 옷을 입었는지 여부 판단
    threshold_pixel_count = 2000  # 적절한 임계값 설정
    return color_pixel_count #> threshold_pixel_count

# 유사도 검증
def calculate_image_similarity(image1, image2):
    # 이미지 크기 맞추기
    # image1, image2 = resize_images(image1, image2)

    # 이미지를 그레이스케일로 변환
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray',gray_image1)
    # cv2.imshow('Gray' ,gray_image2)

    # 이미지 유사도 계산
    result = cv2.matchTemplate(gray_image1, gray_image2, cv2.TM_CCOEFF_NORMED)
    similarity = np.max(result)

    return similarity

# define
target_id = 1
dupli_frame = []
threshold = 0.2

target_shirt_color = [122, 109, 116]
shirt_color_range = 20

target_pants_color = [62, 54, 58]
pants_color_range = 20

target_black_color = [14, 14, 20]
black_color_range = 20

# CSV 파일에서 데이터 읽기
csv_file_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/output_second.csv'  # 본인의 CSV 파일 경로로 바꿔주세요
new_csv_file_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/from_csv_to_bbox.csv'
df = pd.read_csv(csv_file_path)

# 동영상 파일 경로
video_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/data/joo_5min.mp4'  # 본인의 동영상 파일 경로로 바꿔주세요

output_video_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/outputs/output_for_cdist.mp4'
output_captures_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/dupli_id_one/'
overlap_captures_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/overlap'
non_overlap_captures_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/non_overlap'
non_similar_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/non_similar_info.txt'
# case_4_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/case_4.txt'

# 동영상 불러오기
# try:
#     cap = cv2.VideoCapture(int(video_path))
# except:
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not Open :", video_path)
    exit(0)

# 동영상 프레임 크기 및 FPS 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 동영상 저장을 위한 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 각 객체에 대한 색상 매핑을 위한 딕셔너리 생성
color_mapping = {}
non_similar_frame = []
# case_4 = []
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    # print(ret, frame)

    # 동영상 끝에 도달하면 종료
    if not ret:
        break

    # 현재 프레임의 정보 얻기
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_data = df[df['Frame'] == frame_number]
    
    
    duplicated_object_ids = frame_data[frame_data.duplicated('Object ID', keep=False)]['Object ID'].tolist()
    #print(duplicated_object_ids) # 프레임마다[1,1] 이런 식으로 나옴
    #bbox_info = data['xmin']  #data[['xmin', 'ymin', 'xmax', 'ymax']].values
    duplicated_indices = frame_data[frame_data['Object ID'].isin(duplicated_object_ids)].index.tolist()
    # print(duplicated_indices)
    
    a= 0
    if len(duplicated_object_ids) > 1: # 중복 id가 있는 프레임인 경우(다 1이긴 해)
        
        data = df[(df['Frame'] == frame_number) & (df['Object ID'] == target_id)] # 1인 data
        # print(data['xmin'].values, data['xmax'].values)
        bounding_boxes = data[['xmin', 'ymin', 'xmax', 'ymax']].values
        index = data.index.tolist()
        # print(index)
        
        num_objects = bounding_boxes.shape[0] # object 개수
        for i in range(num_objects - 1):
            for j in range(i + 1, num_objects):
                bbox_a = bounding_boxes[i]
                bbox_b = bounding_boxes[j]
                
                # 두 객체의 bounding box가 겹치는지 여부 확인
                overlap = (bbox_a[:2] < bbox_b[2:]) & (bbox_a[2:] > bbox_b[:2])
                
                # 중복되는 애들 다 유사도 검사
                captured_image_a = frame[bbox_a[1]:bbox_a[3], bbox_a[0]:bbox_a[2]]
                captured_image_b = frame[bbox_b[1]:bbox_b[3], bbox_b[0]:bbox_b[2]]
                
                
                if captured_image_a.shape[0] < 100: # 뒤에 지나가는 사람 1번으로 인식한 것 처리->6번 케이스
                    df = df.drop(index=index[0]) 
                    df.to_csv(new_csv_file_path, index=False)
                    continue
                elif captured_image_b.shape[0] < 100:
                    df = df.drop(index=index[1])
                    df.to_csv(new_csv_file_path, index=False)
                    continue
                
                # resize
                captured_image_a = cv2.resize(captured_image_a, (300, 600))
                captured_image_b = cv2.resize(captured_image_b, (300, 600))
                similarity = calculate_image_similarity(captured_image_a, captured_image_b)
                print(frame_number, ': ',similarity)
                
                if similarity > threshold: # 같은 이미지 일때 3번 케이스
                    # captured_image_a의 정보는 남기고, captured_image_b의 정보는 삭제
                    df = df.drop(index=min(duplicated_indices))
                    df.to_csv(new_csv_file_path, index=False)
                    continue
                    
                    
                # else: # 다른 이미지일 때 -> 반대 비디오 체크(따로 프레임, bbox정보 저장) 1,2,4 케이스 -? 밑에 if문으로 구분
                    # captured_image_b의 프레임과 바운딩 박스 정보를 따로 저장
                    # non_similar_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/non_similar_info.txt'  # 원하는 경로로 바꿔주세요

                # 겹치는지 여부 출력
                if np.all(overlap): # 겹치는데 유사하지 않은 애들 -> 양쪽 비디오 보자: 1,2 번 케이스 -> 후처리
                    overlap_capture_a = os.path.join(overlap_captures_path, f"frame_{frame_number}_captured_object_a.png")
                    overlap_capture_b = os.path.join(overlap_captures_path, f"frame_{frame_number}_captured_object_b.png")
                    
                    # # 이미지를 저장하거나 다른 작업을 수행할 수 있음
                    cv2.imwrite(overlap_capture_a, captured_image_a)
                    cv2.imwrite(overlap_capture_b, captured_image_b)
                    
                    non_similar_frame.append(frame_number)
                    print(non_similar_frame)
                else: # 중복 1이 있는데, 안 겹치면 -> 둘 다 1로 인식한 애들: 4번 케이스                
                    # 얘네들은 누가 더 joohyung인지 누가 더 joohyung True에 영향을 주었는지 판단해서 더 joohyung만 살림
                    
                    non_overlap_capture_a = os.path.join(non_overlap_captures_path, f"frame_{frame_number}_captured_object_a.png")
                    non_overlap_capture_b = os.path.join(non_overlap_captures_path, f"frame_{frame_number}_captured_object_b.png")
                    
                    # case_4.append(frame_number)
                    
                    if is_person_wearing_black_color(captured_image_a, target_black_color, black_color_range) > is_person_wearing_black_color(captured_image_b, target_black_color, black_color_range): 
                        # a가 많으면 a가 검정 옷
                        df = df.drop(index=index[0]) 
                        df.to_csv(new_csv_file_path, index=False)
                        continue
                    elif is_person_wearing_black_color(captured_image_a, target_black_color, black_color_range) < is_person_wearing_black_color(captured_image_b, target_black_color, black_color_range): 
                        # a가 많으면 a가 검정 옷
                        df = df.drop(index=index[1]) 
                        df.to_csv(new_csv_file_path, index=False)
                        continue
                    
                    # cv2.imwrite(non_overlap_capture_a, captured_image_a)
                    # cv2.imwrite(non_overlap_capture_b, captured_image_b)
        # overlap = np.any((bbox_array[:, None, :2] <= bbox_array[:, :2]) & (bbox_array[:, None, 2:] >= bbox_array[:, 2:]), axis=-1)
        # if np.sum(overlap) > 1:
        #     print("겹치는 객체가 있습니다.", overlap)
        
        # 중복된 Object ID가 있다면 파일명에 추가적인 숫자를 붙여줌
        # suffix = a
        # a+=1
        # captured_image = frame[ymin:ymax, xmin:xmax]
        # capture_path = os.path.join(output_captures_path, f"capture_frame_{frame_number}_id_{target_id}_suffix_{suffix}.png")
        # cv2.imwrite(capture_path, captured_image)
        # print(f"Captured Frame: {frame_number}, Object ID: {target_id}. Saved at: {capture_path}")
    
    # bounding box 그리기
    frame_data = df[df['Frame'] == frame_number]
    for index, row in frame_data.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        object_id = row['Object ID']  # "Object ID" 정보 가져오기

        # 객체별로 고유한 색상 할당
        if object_id not in color_mapping:
            color_mapping[object_id] = tuple(np.random.randint(0, 255, 3).tolist())

        color = color_mapping[object_id]


        # 중앙을 기준으로 오른쪽에 위치한 경우 ID를 999로 변경 -> object_tracker에 작성함
        # if object_id == target_id and (xmin + xmax) / 2 > (width / 2):
        #     object_id = 999
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f'ID: {object_id}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # 프레임 번호 표시
    cv2.putText(frame, f'Frame: {frame_number}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
        


    # 동영상 파일에 프레임 쓰기
    out.write(frame)

    # 화면에 표시
    cv2.imshow('Video with Bounding Boxes', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 종료
with open(non_similar_path, 'w') as file:
    for item in non_similar_frame:
        file.write(f'{item}\n')
        
# with open(case_4_path, 'w') as file:
#     for item in case_4:
#         file.write(f'{item}\n')
cap.release()
out.release()
cv2.destroyAllWindows()
