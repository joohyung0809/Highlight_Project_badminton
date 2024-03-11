# joo-> seo 인 부분만 체크해서 joo로 만들기

import cv2
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


# define

flag = 0
target_id = 1
previous_data = None


def find_closest_id(target_id, current_frame_data, previous_frame_data):
    # 현재 프레임에서 사라진 id에 대한 정보 찾기
    missing_id_info = [obj for obj in previous_frame_data if obj['id'] == target_id] # 이전 프레임에서 1번에 속하는 데이터 정보
    # [{'id': 1, 'bbox': '574,606,689,854'}] <- 이런 형태
    # print(missing_id_info)

    if not missing_id_info:
        print(f"ID {target_id} not found in the previous frame.")
        return None

    # 현재 프레임의 다른 객체들 중에서 가장 가까운 id 찾기
    current_frame_ids = [obj['id'] for obj in current_frame_data]
    missing_id_bbox = list(map(int, missing_id_info[0]['bbox'].split(',')))  # 이전 프레임에서 사라진 id의 bounding box 정보
    # print(missing_id_bbox) 
    # [574, 606, 689, 854] <- 이런 형태

    # 현재 프레임에서 가장 가까운 id 찾기
    distances = cdist([missing_id_bbox], [list(map(int, obj['bbox'].split(','))) for obj in current_frame_data]) 
    closest_id_index = np.argmin(distances)

    closest_id = current_frame_ids[closest_id_index]

    # return closest_id
    return closest_id ,closest_id_index


def detect_id_change(current_frame, previous_frame, target_id, threshold=100):
    # print(current_frame)
    # print(previous_frame)
    # 현재 프레임과 이전 프레임의 Object ID가 target_id인 경우, bounding box 변화를 확인
    current_id = current_frame[current_frame['Object ID'] == target_id]
    previous_id = previous_frame[previous_frame['Object ID'] == target_id]

    if len(current_id) == 0 and len(previous_id) > 0: # 이전에는 1이 있는데 현재에는 없어 -> 가장 가까운 걸 1로 할당
        return False  # 현재나 이전 프레임에 해당 ID가 없음
    
    if len(previous_id) == 0:
        print(f"Warning: No previous data found for Object ID {target_id}")
        return False

    current_bbox = current_id[['xmin', 'ymin', 'xmax', 'ymax']].values[0]
    previous_bbox = previous_id[['xmin', 'ymin', 'xmax', 'ymax']].values[0]

    # 현재와 이전 bounding box 간의 거리 계산
    distance = np.linalg.norm(current_bbox - previous_bbox)

    return distance > threshold


# CSV 파일에서 데이터 읽기
csv_file_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/from_after_interpol_csvtobbox.csv'  # 본인의 CSV 파일 경로로 바꿔주세요
df = pd.read_csv(csv_file_path)

csv_update_file_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/from_after_cdist_csvtobbox.csv'
# csv_update_file_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/from_cdisit_csvtobbox_updated.csv' # 원래

# 동영상 파일 경로
video_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/data/joo_5min.mp4'  # 본인의 동영상 파일 경로로 바꿔주세요
output_video_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/outputs/from_after_cdist_csvtobbox.mp4'

# 동영상 불러오기
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
previous_frame_data = None

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 동영상 끝에 도달하면 종료
    if not ret:
        break

    # 현재 프레임의 정보 얻기
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_frame_data = df[df['Frame'] == frame_number]
    
    now_data = [
    {'id': int(row['Object ID']), 'bbox': f"{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}"}
    for index, row in current_frame_data.iterrows()
    ]

    # print(f"before: {previous_frame_data}")
    # print(f"now: {now_data}")

    # 첫 프레임이 아닌 경우에만 ID 변경 여부를 체크
    if previous_frame_data is not None:
        # ID 변경 여부 감지
        if detect_id_change(current_frame_data, previous_frame_data, target_id):
            # print(f"Object ID {target_id} changed or disappeared at frame {frame_number}")
            
    
            # print(f'{frame_number}: change moment(joo->seo)')
            closest_id, closest_index = find_closest_id(target_id, now_data, before_data) # 수정해서 한 프레임 안에서의 인덱스를 출력하게 만듦
            # closest_id, closest_index = find_closest_id(target_id, now_data, before_data, width)
            print(f"frame: {frame_number}, closest_id: {closest_id}, closest_index: {closest_index}")
            if closest_id: # csv 갱신하자 / current_data 갱신
                # now_data = [{'id': target_id, 'bbox': obj['bbox']} if obj['id'] == closest_id else obj for obj in now_data]
                if closest_id == 1: # 1이 다른 곳으로 순간이동 했고, 그때 closest_id가 joohyung으로 판별 하지 않은 경우
                    df.loc[(df['Frame'] == frame_number) & (df['Object ID'] == closest_id), 'Object ID'] = 999  # df에 진짜 index로 접근해서 closest_id를 1로 변경
                    df.to_csv(csv_update_file_path, index=False)
                    continue
                else: # 순간이동 했는데, 그때 closest_id가 joohyung으로 판별하는 경우
                    closest_index_in_df = current_frame_data.iloc[closest_index].name
                    # obj_bbox = df.at[closest_index_in_df, 'xmin']  #list(map(int, obj['bbox'].split(',')))
                    # obj_center_x = (df.at[closest_index_in_df, 'xmin'] + df.at[closest_index_in_df, 'xmax']) / 2
                    # if (obj_center_x > width / 2) and frame_number>7000:
                    #     continue
                    # closest_index_in_df에 해당하는 행의 Object ID를 target_id로 변경
                        # 현재 프레임에서 1번 Object ID와 closest_id 번호의 Object ID를 서로 바꾸기
                    df.loc[(df['Frame'] == frame_number) & (df['Object ID'] == target_id), 'Object ID'] = closest_id
                    df.at[closest_index_in_df, 'Object ID'] = target_id  # df에 진짜 index로 접근해서 closest_id를 1로 변경
                    df.to_csv(csv_update_file_path, index=False)
        else: # 이전에는 1이 있지만 현재 1이 없을 때
            pass
            
                        
            
                    
        # 현재 프레임 정보를 다음 반복에서 사용하기 위해 저장
        # previous_frame_data = df[df['Frame'] == frame_number] # 여기까지 오면 df 갱신 됐고, 다시 df 받아오면 됨
        # before_data = [ # previous가 이곳 한정으로 current_frame_data가 되었으니까 사용
        # {'id': int(row['Object ID']), 'bbox': f"{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}"}
        # for index, row in previous_frame_data.iterrows()
        # ]
        current_frame_data = df[df['Frame'] == frame_number]
        if detect_id_change(current_frame_data, previous_frame_data, target_id): # 갱신된 데이터로 다시 detect 해보자. 걸리면 1이 없다는 뜻. -> 현재 1을 999로 돌리고 continue
            df.loc[(df['Frame'] == frame_number) & (df['Object ID'] == target_id), 'Object ID']=999
            df.to_csv(csv_update_file_path, index=False)
            continue
   
        
    # bounding box 그리기
    current_frame_data = df[df['Frame'] == frame_number]
    
    for index, row in current_frame_data[current_frame_data['Object ID'] == target_id].iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        object_id = row['Object ID']

        # 객체별로 고유한 색상 할당
        if object_id not in color_mapping:
            color_mapping[object_id] = tuple(np.random.randint(0, 255, 3).tolist())

        color = color_mapping[object_id]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f'ID: {object_id}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    cv2.LINE_AA)

    # 프레임 번호 표시
    font_size = 2
    font_thickness = 3
    font_color = (255, 255, 255)
    font_position = (10, int(height - 5 / 30 * height))

    cv2.putText(frame, f'Frame: {frame_number}', font_position, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color,
                font_thickness, cv2.LINE_AA)
    
    
    
    previous_frame_data = df[df['Frame'] == frame_number] # 여기까지 오면 df 갱신 됐고, 다시 df 받아오면 됨
    before_data = [ # previous가 이곳 한정으로 current_frame_data가 되었으니까 사용
    {'id': int(row['Object ID']), 'bbox': f"{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}"}
    for index, row in previous_frame_data.iterrows()
    ]
    
    # print(f"previous(take df[frame_number]): {previous_frame_data}, before :{before_data}")
    

    # 동영상 파일에 프레임 쓰기
    out.write(frame)

    # 화면에 표시
    cv2.imshow('Video with Bounding Boxes', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 종료
cap.release()
out.release()
cv2.destroyAllWindows()

