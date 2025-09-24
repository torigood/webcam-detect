## 환경 세팅

# 1. 가상환경 만들기 (venv)

python -m venv .venv
전역이 아닌 venv 가상 환경에만 설치하기 위한 작업

# 2. 가상환경 켜기

.\.venv\Scripts\Activate.ps1

(.venv) PS 파일 위치 > 이런식으로 뜨면 성공

# 3. 패키지 설치

pip install --upgrade pip
pip install ultralytics opencv-python numpy onnxruntime torch torchvision torchaudio

# 4. scripts/verify_setup.py 파일 만들어서 아래 코드 붙여넣기:

1. import cv2, torch, ultralytics
2. print("OpenCV:", cv2.**version**)
3. print("PyTorch:", torch.**version**, "CUDA available:", torch.cuda.is_available())
4. print("Ultralytics:", ultralytics.**version**)

5. cap = cv2.VideoCapture(0)
6. ok, \_ = cap.read()
7. cap.release()
8. print("Webcam:", "OK" if ok else "FAILED")

# 5. 터미널에서 실행 해서 Webcam 확인

python scripts/verify_setup.py

Webcam: OK 가 나와야함

하는 일:

1. OpenCV / PyTorch / Ultralytics 버전 출력
2. CUDA(GPU) 사용 가능 여부 확인
3. 웹캠이 연결되고 읽히는지 확인
4. 실행하면 → 환경이 잘 갖춰졌는지, 카메라가 되는지 바로 알 수 있음.

# 6. AI 관련 라이브러리 설치

pip install ultralytics opencv-python numpy onnxruntime torch torchvision torchaudio

1. ultralytics → YOLO 모델 쉽게 쓰는 라이브러리
2. opencv-python → 영상 처리, 카메라 입력
3. numpy → 수학 연산 필수 도구
4. onnxruntime → 모델 빠르게 실행하는 러너
5. torch/torchvision/torchaudio → 딥러닝 엔진(PyTorch)

# webcam_detect.py

python scripts/webcam_detect.py

이런식으로 실행도 가능 python scripts/webcam_detect.py --imgsz 416 --conf 0.3

왜 이렇게 하느냐

1. 사전학습 YOLOv8(nano)
   COCO 80클래스에 대해 이미 학습됨 → 내 데이터 없이 즉시 사용 가능.
   yolov8n.pt는 가장 가벼워서 CPU에서도 돌아가기 쉬움(지금 PyTorch CPU).

2. imgsz를 480으로 시작
   해상도가 낮을수록 추론 속도↑. CPU 환경에서 프레임레이트 확보에 유리.
   필요하면 416/384로 더 낮춰 속도를 얻고, 정확도가 필요하면 544/640으로 올리면 됨(대신 느려짐).

3. conf=0.5 기본
   신뢰도가 낮은 박스는 걸러서 오탐(허상) 줄이기.
   작은 물체가 잘 안 잡히면 0.3~0.4로 낮춰 재시도.

4. CPU 우선
   지금 환경이 CUDA available: False.
   GPU 없이도 “객체 인식의 개념/파이프라인”을 익히는 데 충분. 나중에 GPU 붙이면 같은 코드로 속도만 상승.

5. 요약
   YOLOv8 사전학습 모델(yolov8n.pt) = 뇌 (사물 보는 AI)
   OpenCV(cv2.VideoCapture) = 눈 (카메라에서 영상 가져옴)
   results[0].plot() = 손 (박스+라벨을 그려줌)
   cv2.imshow() = 화면 (결과 보여줌)
