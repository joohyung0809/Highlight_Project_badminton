# after_interpol_csvtobbox 돌리고 나온 csv 파일에서 중간에 1번 없으면 채워넣기 -> detect 후 change id

import pandas as pd
import numpy as np
import cv2
from scipy.spatial.distance import cdist



def find_closest_id(target_id, current_frame_data, previous_frame_data):
    # print(target_id)
    # 현재 프레임에서 사라진 id에 대한 정보 찾기
    missing_id_info = [obj for obj in previous_frame_data if obj['id'] == target_id] # 이전 프레임에서 1번에 속하는 데이터 정보
    

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
    # closest_distance = distances[0, closest_id_index]


    closest_id = current_frame_ids[closest_id_index]

    # return closest_id
    return closest_id ,closest_id_index



# CSV 파일 읽기
df = pd.read_csv("C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/for7606_from_cdisit_csvtobbox_updated.csv") # 진짜///
# df = pd.read_csv("C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/from_after_interpol_csvtobbox.csv")
# df = pd.read_csv("C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/from_after_cdist_csvtobbox.csv")

# Object ID가 1인 행만 추출
id_1_data = df[df['Object ID'] == 1]

# 모든 프레임 번호 가져오기
all_frames = df['Frame'].unique()

# 1번 ID가 없는 연속된 프레임 찾기
target_id = 1
result = []
current_sequence = []
last_detecting = []
flag = 0 # 짝수일 때는 joo-> seo / 홀수일 때는 seo->joo


for frame in all_frames:
    if frame not in id_1_data['Frame'].tolist():
        current_sequence.append(frame)
    else:
        if current_sequence:
            result.append(current_sequence)
            current_sequence = []

# 마지막으로 끝나는 경우를 확인
if current_sequence:
    result.append(current_sequence)

frames_to_update = []


# prev_bbox와 next_bbox에 대한 코드 추가
for bundle in result:
    prev_frame = bundle[0] - 1
    next_frame = bundle[-1] + 1
    current_frame_data = df[df['Frame'] == prev_frame+1] 
    
    
    # 이전 프레임의 bounding box 정보 가져오기
    prev_bbox = id_1_data.loc[id_1_data['Frame'] == prev_frame, ['xmin', 'ymin', 'xmax', 'ymax']].values[0]

    # 다음 프레임의 bounding box 정보 가져오기
    next_bbox = id_1_data.loc[id_1_data['Frame'] == next_frame, ['xmin', 'ymin', 'xmax', 'ymax']].values[0]
    
    # bounding box가 화면 밖으로 나가는 경우를 확인
    prev_x_center = (prev_bbox[0] + prev_bbox[2]) / 2
    next_x_center = (next_bbox[0] + next_bbox[2]) / 2
    width = next_bbox[2] - next_bbox[0]
    
    # 이전과 다음 bounding box의 거리 계산
    distance = np.linalg.norm(next_bbox - prev_bbox)
    if distance >= 350: # 350 넘은 애들 3개
        print(f"frame: {prev_frame},{next_frame}, distance : {distance}")
        
        if ((prev_x_center < 0 and next_x_center < 0) or (prev_x_center > width and next_x_center > width)): # 밖으로 나간 애
            if (flag % 2) == 0:# joo->seo
                last_detecting.append([prev_frame, next_frame-1])
                # print(f"frame: {prev_frame},{next_frame}, distance : {distance}")
                flag += 1
                continue
        else: # 밖으로 나간 애
            continue
        
    
    for frame in bundle:
        num_bundle = len(bundle)
        filled_bbox = (prev_bbox + (next_bbox - prev_bbox) * (frame - bundle[0]) / num_bundle).astype(int)

        frames_to_update.append([frame, 1, 'person'] + filled_bbox.tolist())
        
    

df_to_update = pd.DataFrame(frames_to_update, columns=['Frame', 'Object ID', 'Class', 'xmin', 'ymin', 'xmax', 'ymax'])
df = pd.concat([df, df_to_update]).sort_values(by=['Frame', 'Object ID']).reset_index(drop=True)

df.to_csv("C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/no180pixel_output_after_cdist_updated.csv", index=False)
print(last_detecting)



df = pd.read_csv("C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/no180pixel_output_after_cdist_updated.csv")
csv_update_file_path = ("C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/no180pixel_output_after_cdist_final.csv")

for frame in last_detecting:
    # 작업 수행을 위한 범위 설정
    
    previous_data = None
    for frame_number in range(frame[0], frame[1]+1):
        current_data = df[df['Frame']== frame_number]
        now_data = [
        {'id': int(row['Object ID']), 'bbox': f"{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}"}
        for index, row in current_data.iterrows()
        ]
        
        if previous_data is not None: # 여기서 거리 탐색
            # print(f"previous: {previous_data}")
            closest_id, closest_index = find_closest_id(target_id, now_data, before_data) # 수정해서 한 프레임 안에서의 인덱스를 출력하게 만듦
            # print(closest_id, closest_index)
            closest_index_in_df = current_data.iloc[closest_index].name
            # print(closest_index_in_df)
            # df.loc[(df['Frame'] == frame_number) & (df['Object ID'] == target_id), 'Object ID'] = closest_id
            df.at[closest_index_in_df, 'Object ID'] = target_id  # df에 진짜 index로 접근해서 closest_id를 1로 변경
            df.to_csv(csv_update_file_path, index=False)
            
        
        
        # 이전에 df 갱신 시켜야 함
        previous_data = df[df['Frame'] == frame_number] # 여기까지 오면 df 갱신 됐고, 다시 df 받아오면 됨
        
        before_data = [ # previous가 이곳 한정으로 current_frame_data가 되었으니까 사용
        {'id': int(row['Object ID']), 'bbox': f"{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}"}
        for index, row in previous_data.iterrows()
        ]
        
        
        

df = pd.read_csv("C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/no180pixel_output_after_cdist_final.csv")
csv_update_file_path = ("C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/no180pixel_output_after_cdist_realfinal.csv")
        
        
target_id = 1
frame_number = 2
previous_frame_data = None

def detect_id_change(current_frame, previous_frame, target_id, threshold=100):
    # 현재 프레임과 이전 프레임의 Object ID가 target_id인 경우, bounding box 변화를 확인
    current_id = current_frame[current_frame['Object ID'] == target_id]
    previous_id = previous_frame[previous_frame['Object ID'] == target_id]
    # print(f"current_id: {current_id}, previous id: {previous_id}")

    if len(current_id) == 0 or len(previous_id) == 0: # 이전에는 1이 있는데 현재에는 없어 -> 가장 가까운 걸 1로 할당
        return False  # 현재나 이전 프레임에 해당 ID가 없음

    current_bbox = current_id[['xmin', 'ymin', 'xmax', 'ymax']].values[0]
    previous_bbox = previous_id[['xmin', 'ymin', 'xmax', 'ymax']].values[0]

    # 현재와 이전 bounding box 간의 거리 계산
    distance = np.linalg.norm(current_bbox - previous_bbox)
    # print(distance)


    return distance > threshold

for i in range(1, 10351):
    current_frame_data = df[df['Frame'] == i]
    now_data = [
        {'id': int(row['Object ID']), 'bbox': f"{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}"}
        for index, row in current_frame_data.iterrows()
    ]
    
    if previous_frame_data is not None:
        if detect_id_change(current_frame_data, previous_frame_data, target_id): # 이때 가까운 애들 찾기
            closest_id, closest_index = find_closest_id(target_id, now_data, before_data) # 수정해서 한 프레임 안에서의 인덱스를 출력하게 만듦
            print(closest_id, closest_index)
            closest_index_in_df = current_frame_data.iloc[closest_index].name
            print(closest_index_in_df)
            df.loc[(df['Frame'] == i) & (df['Object ID'] == target_id), 'Object ID'] = closest_id
            df.at[closest_index_in_df, 'Object ID'] = target_id  # df에 진짜 index로 접근해서 closest_id를 1로 변경
            df.to_csv(csv_update_file_path, index=False)
            print(i)
    
    
    previous_frame_data = df[df['Frame'] == i] # 여기까지 오면 df 갱신 됐고, 다시 df 받아오면 됨
    before_data = [ # previous가 이곳 한정으로 current_frame_data가 되었으니까 사용
        {'id': int(row['Object ID']), 'bbox': f"{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}"}
        for index, row in previous_frame_data.iterrows()
        ]







