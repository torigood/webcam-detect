import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 윈도우면 CAP_DSHOW 권장
    if cap.isOpened():
        ok, frame = cap.read()
        print(f"Index {i}: Opened={ok}")
        cap.release()
    else:
        print(f"Index {i}: Failed")


# python scripts/check_cameras.py 
# 카메라 인덱스 0~4까지 시도해서 연결되는지 확인 
# 이걸로 카메라 인덱스 확인 후, scripts/webcam_detect.py 에 반영