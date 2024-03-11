import cv2
import numpy as np

# def resize_images(image1, image2): # 애초에 저장할 때 부터 크기를 정해서 저장하자
#     # 두 이미지의 크기를 비교하여 크기를 맞춰줌
#     h1, w1, _ = image1.shape
#     h2, w2, _ = image2.shape

#     if h1 != h2 or w1 != w2:
#         image2 = cv2.resize(image2, (w1, h1))

#     return image1, image2

def calculate_image_similarity(image1, image2):
    # 이미지 크기 맞추기
    # image1, image2 = resize_images(image1, image2)

    # 이미지를 그레이스케일로 변환
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray',gray_image1)
    # cv2.imshow('Gray' ,gray_image2)

    # 이미지 유사도 계산
    result = cv2.matchTemplate(gray_image1, gray_image2, cv2.TM_CCOEFF_NORMED)
    similarity = np.max(result)

    return similarity

# 예시 이미지 로드
image_path1 = "C:/Users/joohy/Computer_Vision/yolov4-deepsort/overlap/frame_9783_captured_object_a.png"
image_path2 = "C:/Users/joohy/Computer_Vision/yolov4-deepsort/overlap/frame_9783_captured_object_b.png"

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# 이미지 유사도 계산
similarity = calculate_image_similarity(image1, image2)

# 유사도 출력
print(f"이미지 유사도: {similarity}")

# 임계값을 설정하여 유사도에 따라 판단할 수 있음
threshold = 0.7  # 예시 임계값
if similarity > threshold:
    print("두 이미지는 유사합니다.")
else:
    print("두 이미지는 유사하지 않습니다.")
    
cv2.waitKey(0)
cv2.destroyAllWindows()
