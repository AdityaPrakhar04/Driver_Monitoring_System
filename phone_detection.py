from ultralytics import YOLO
import cv2
model=YOLO("runs/detect/runs/phone/yolov8n_phone-2/weights/best.pt")
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        print("Could not access webcam.")
        break
    frame=cv2.flip(frame,1)
    results=model(frame,conf=0.30)
    annotated_frame=results[0].plot()
    cv2.imshow("DMS - Phone Detection",annotated_frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()