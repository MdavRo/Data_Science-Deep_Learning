from ultralytics import YOLO
import cv2
import torch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
model = YOLO(r"C:\Users\User\Downloads\best.pt").to(device)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Perform object detection using the YOLO model
    results = model.predict(frame)

    cv2.imshow('Object Detection', results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
