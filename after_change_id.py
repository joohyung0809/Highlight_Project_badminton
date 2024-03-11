# import cv2
# import numpy as np
# import pandas as pd

# # CSV 파일 불러오기
# csv_file_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/no180pixel_output_after_cdist_realfinal.csv'
# df = pd.read_csv(csv_file_path)
# df = df[df['Object ID'] == 1]

# # 동영상 파일 경로
# video_file_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/data/joo_5min.mp4'
# cap = cv2.VideoCapture(video_file_path)

# # 캡쳐된 프레임을 저장할 디렉토리
# output_directory = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/jump/'

# # 캡쳐할 프레임 정보를 담을 리스트
# captured_frames = []

# # 각 프레임을 순회하면서 점프 여부 확인
# for i in range(30, len(df)):
#     current_frame = df.iloc[i]
#     previous_frame = df.iloc[i - 30]

#     # 현재 프레임과 이전 프레임의 bounding box 정보 추출
#     current_bbox = np.array([int(current_frame['xmin']), int(current_frame['ymin']),
#                              int(current_frame['xmax']), int(current_frame['ymax'])])
    
#     previous_bbox = np.array([int(previous_frame['xmin']), int(previous_frame['ymin']),
#                               int(previous_frame['xmax']), int(previous_frame['ymax'])])

#     # 두 bounding box 사이의 거리 측정
#     distance = np.linalg.norm(current_bbox - previous_bbox)

#     # 거리가 threshold 이상이면 캡쳐
#     if distance >= 100:  # 원하는 threshold 값으로 조절
#         print(f"frame: {i} ,distance: {distance}")
#         captured_frames.append(current_frame)

#         # 현재 프레임에서 bounding box 영역만 추출
#         xmin, ymin, xmax, ymax = current_bbox
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()

#         if ret:
#             cropped_frame = frame[ymin:ymax, xmin:xmax]

#             # 캡쳐된 이미지 저장
#             image_name = f"{output_directory}jump_capture_{i}.png"
#             cv2.imwrite(image_name, cropped_frame)

# print("캡쳐된 프레임 정보:")
# print(pd.DataFrame(captured_frames))

# # 동영상 캡쳐 완료 후 해제
# cap.release()
# cv2.destroyAllWindows()
