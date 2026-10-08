import cv2
import mediapipe as mp
BaseOptions=mp.tasks.BaseOptions
FaceLandmarker=mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions=mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode=mp.tasks.vision.RunningMode
options=FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path="face_landmarker.task"),running_mode=VisionRunningMode.IMAGE,num_faces=1)
landmarker=FaceLandmarker.create_from_options(options)
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        print("Unable to access camera")
        break
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
    results=landmarker.detect(mp_image)
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            h,w,_=frame.shape
            for landmark in face_landmarks:
                x=int(landmark.x*w)
                y=int(landmark.y*h)
                cv2.circle(
                    frame,
                    (x,y),
                    1,
                    (0,255,0),
                    -1
                )
    cv2.imshow("MediaPipe Face Landmarks",frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break
cap.release()
landmarker.close()
cv2.destroyAllWindows()