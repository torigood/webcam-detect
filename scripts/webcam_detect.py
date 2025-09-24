import argparse, cv2
from ultralytics import YOLO

def main(model_path: str, source: str, imgsz: int, conf: float):
    model = YOLO(model_path)                     # COCO 80클래스 사전학습
    cap = cv2.VideoCapture(0 if source=="0" else source, cv2.CAP_ANY) # 0=기본웹캠 1=외장웹캠
    if not cap.isOpened(): raise RuntimeError("Webcam open failed")

    # CPU 속도 대비 해상도 낮춰서 시작 (필요시 조절)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    while True:
        ok, frame = cap.read()
        if not ok: break
        # 한 장면(프레임)만 모델에 넣어서 추론
        results = model.predict(frame, imgsz=imgsz, conf=conf, device="cpu", verbose=False)
        annotated = results[0].plot()            # 박스/라벨 그린 화면

        cv2.imshow("Webcam Object Detection (YOLOv8)", annotated)
        if cv2.waitKey(1) & 0xFF == 27:         # ESC로 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")   # 가장 가벼운 nano
    ap.add_argument("--source", default="0")           # 0=기본 웹캠 1=외장 웹캠
    ap.add_argument("--imgsz", type=int, default=480)  # CPU면 480 권장(느리면 416/384)
    ap.add_argument("--conf", type=float, default=0.5) # 신뢰도 임계값
    args = ap.parse_args()
    main(args.model, args.source, args.imgsz, args.conf)
