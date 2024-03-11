import os
import pandas as pd
import cv2
import imageio


def get_file_names(directory):
    file_names = []
    for filename in os.listdir(directory):
        name, extension = os.path.splitext(filename)
        file_names.append(int(name))  # 문자열을 정수로 변환해서 추가
    
    return file_names

def grouping (files):
    result = []
    current = []

    for number in files:
        if not current or number == current[-1] + 1:
            current.append(number)
        else:
            result.append(current)
            current = [number]

    if current:
        result.append(current)

    return result

# 사용 예시
directory_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/jump'
files = get_file_names(directory_path)
# print(files)

grouped = grouping(files)

grouped = [group for group in grouped if len(group) > 1]

grouped = [round(sum(group) / len(group)) for group in grouped]


print(grouped)

# 비디오 경로
video_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/data/joo_5min.mp4'

save_directory = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/for_short_video/'


# CSV 파일 읽기
df = pd.read_csv('C:/Users/joohy/Computer_Vision/yolov4-deepsort/save_file/no180pixel_output_after_cdist_realfinal.csv')
df = df[df['Object ID'] == 1]

# 반복문을 통해 각 그룹에 대해 처리
for frame in grouped:
    for current_frame_number in range(frame - 30, frame + 31):
        # 현재 프레임에 해당하는 데이터 추출
        current_frame_data = df[df['Frame'] == current_frame_number]

        # 캡쳐를 수행할 때, bounding box 정보를 이용하여 캡쳐 로직을 작성
        for index, current_frame in current_frame_data.iterrows():
            # print(f"Frame: {current_frame_number}, Bounding Box: {current_frame['BoundingBox']}")

            # 여기에 실제로 캡쳐하는 로직을 작성하면 됨
            # 캡쳐를 수행할 동영상 로드
            cap = cv2.VideoCapture(video_path)

            # 원하는 프레임으로 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)

            # 프레임 읽기
            ret, original_frame = cap.read()

            xmin, ymin, xmax, ymax = (
                max(int(current_frame['xmin']) - 40, 0),
                max(int(current_frame['ymin']) - 50, 0),
                int(current_frame['xmax']) + 25,
                int(current_frame['ymax']) + 35
            )

            # 이미지 자르기
            cropped_frame = original_frame[ymin:ymax, xmin:xmax]
            
            # 원하는 해상도로 이미지 크기 조정
            target_resolution = (1920, 1080)  # 예시: Full HD
            cropped_frame_resized = cv2.resize(cropped_frame, target_resolution)

            # 이미지 저장
            save_path = os.path.join(save_directory, f"{current_frame_number}.png")
            cv2.imwrite(save_path, cropped_frame)

            # 캡쳐 동작 후 동영상 객체 해제
            cap.release()
            
            

# 이미지 파일 경로와 출력 동영상 파일명 설정
image_path_template = "C:/Users/joohy/Computer_Vision/yolov4-deepsort/for_short_video/{}.png"
output_video_path = "C:/Users/joohy/Computer_Vision/yolov4-deepsort/for_short_video/output{}.mp4"


# 이미지를 읽어서 동영상에 추가
for number, i in enumerate(grouped):
    # 프레임 범위 설정
    start_frame = i - 30
    end_frame = i + 30

    first_frame_path = image_path_template.format(start_frame)
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # 비디오 코덱 및 프레임 속도 설정
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 20  # 프레임 속도 (예: 30fps)

    # 비디오 작성자 객체 생성
    video_writer = cv2.VideoWriter(output_video_path.format(number), fourcc, fps, (width, height))

    for frame_number in range(start_frame, end_frame + 1):
        image_path = image_path_template.format(frame_number)
        frame = cv2.imread(image_path)

        # 이미지 크기가 다르다면 첫 번째 프레임 크기로 조정
        if frame.shape != (height, width, 3):
            frame = cv2.resize(frame, (width, height))

        # 동영상에 프레임 추가
        video_writer.write(frame)

# 비디오 작성자 객체 종료
video_writer.release()
    