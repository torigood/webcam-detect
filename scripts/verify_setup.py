import cv2, torch, ultralytics
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("Ultralytics:", ultralytics.__version__)

cap = cv2.VideoCapture(0)
ok, _ = cap.read()
cap.release()
print("Webcam:", "OK" if ok else "FAILED")
