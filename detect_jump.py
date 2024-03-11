import cv2
import pandas as pd

# CSV 파일 불러오기
csv_file_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/no180pixel_output_after_cdist_realfinal.csv'
df = pd.read_csv(csv_file_path)
df = df[df['Object ID'] == 1]

# 동영상 파일 경로
video_file_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/data/joo_5min.mp4'
cap = cv2.VideoCapture(video_file_path)

# 캡쳐된 프레임을 저장할 디렉토리
output_directory = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/jump/'

# 캡쳐할 프레임 정보를 담을 리스트
captured_frames = []

# 각 프레임을 순회하면서 점프 여부 확인
for i in range(30, len(df)):
    # print(i)
    current_frame = df.iloc[i]
    # print(current_frame)
    previous_frame = df.iloc[i - 30]

    # ymax 비교
    ymax_diff = current_frame['ymax'] - previous_frame['ymax']
    xmin_diff = current_frame['xmin'] - previous_frame['xmin']
    xmax_diff = current_frame['xmax'] - previous_frame['xmax']

    # A케이스가 목적 or B 케이스가 목적 or C케이스가 목적
    if (ymax_diff > 80) or (ymax_diff < -70) or ((xmin_diff < -65) and (xmax_diff <-65)):
        # 현재 프레임에서 bounding box 정보 추출
        xmin, ymin, xmax, ymax = (
            max(int(current_frame['xmin']) - 15, 0),  # xmin이 음수면 0으로 대체
            max(int(current_frame['ymin']) - 40, 0),  # ymin이 음수면 0으로 대체
            int(current_frame['xmax']) + 15,
            int(current_frame['ymax']) + 20
        )
        # print(f"frame: {current_frame['Frame']}, i: {i}, diff :{ymax_diff}")
        
        captured_frames.append(current_frame['Frame'])

        # 동영상에서 해당 프레임 가져오기
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame['Frame'])
        ret, frame = cap.read()

        if ret:
            # print(i)
            # bounding box 영역만 추출
            # cropped_frame = frame[ymin:ymax, xmin:xmax]

            # 캡쳐된 이미지 저장
            image_name = f"{output_directory}{current_frame['Frame']}.png"
            # cv2.imwrite(image_name, cropped_frame)
            cv2.imwrite(image_name, frame)


captured_frames_df = pd.DataFrame(captured_frames)

# 캡쳐된 프레임 정보를 CSV 파일로 저장
captured_frames_csv_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/jump/captured_frames.csv'
captured_frames_df.to_csv(captured_frames_csv_path, index=False)

print("캡쳐된 프레임 정보:")
print(pd.DataFrame(captured_frames))

# 동영상 캡쳐 완료 후 해제
cap.release()
cv2.destroyAllWindows()
