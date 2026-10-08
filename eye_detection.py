import cv2
import mediapipe as mp
import math
import time
LEFT_EYE=[362,385,387,263,373,380]
RIGHT_EYE=[33,160,158,133,153,144]
MOUTH=[61,81,13,311,291,402,14,178]
EAR_THRESHOLD=0.20
CLOSED_TIME_THRESHOLD=1.5
MAR_THRESHOLD=0.50
YAWN_TIME_THRESHOLD=1.0
def distance(p1,p2):
    return math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)
def calculate_ear(landmarks,eye):
    p0=landmarks[eye[0]]
    p1=landmarks[eye[1]]
    p2=landmarks[eye[2]]
    p3=landmarks[eye[3]]
    p4=landmarks[eye[4]]
    p5=landmarks[eye[5]]
    horizontal=distance(p0,p3)
    vertical_1=distance(p1,p5)
    vertical_2=distance(p2,p4)
    ear=(vertical_1+vertical_2)/(2*horizontal)
    return ear
def calculate_mar(landmarks):
    horizontal=distance(landmarks[MOUTH[0]],landmarks[MOUTH[4]])
    vertical_1=distance(landmarks[MOUTH[1]],landmarks[MOUTH[7]])
    vertical_2=distance(landmarks[MOUTH[2]],landmarks[MOUTH[6]])
    vertical_3=distance(landmarks[MOUTH[3]],landmarks[MOUTH[5]])
    mar=(vertical_1+vertical_2+vertical_3)/(3*horizontal)
    return mar
mp_face_landmarker=mp.tasks.vision.FaceLandmarker
BaseOptions=mp.tasks.BaseOptions
FaceLandmarkerOptions=mp.tasks.vision.FaceLandmarkerOptions
RunningMode=mp.tasks.vision.RunningMode
options=FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path="face_landmarker.task"),running_mode=RunningMode.IMAGE,num_faces=1)
cap=cv2.VideoCapture(0)
closed_start_time=None
yawn_start_time=None
with mp_face_landmarker.create_from_options(options) as landmarker:
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        frame=cv2.flip(frame,1)
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_image=mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        result=landmarker.detect(mp_image)
        if result.face_landmarks:
            landmarks=result.face_landmarks[0]
            left_ear=calculate_ear(landmarks,LEFT_EYE)
            right_ear=calculate_ear(landmarks,RIGHT_EYE)
            ear=(left_ear+right_ear)/2
            if ear<EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time=time.time()
                closed_duration=time.time()-closed_start_time
                if closed_duration>=CLOSED_TIME_THRESHOLD:
                    eye_status="DROWSY"
                else:
                    eye_status="EYES CLOSED"
            else:
                closed_start_time=None
                closed_duration=0
                eye_status="EYES OPEN"
            mar=calculate_mar(landmarks)
            if mar>MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time=time.time()
                yawn_duration=time.time()-yawn_start_time
                if yawn_duration>=YAWN_TIME_THRESHOLD:
                    yawn_status="YAWNING"
                else:
                    yawn_status="MOUTH OPEN"
            else:
                yawn_start_time=None
                yawn_duration=0
                yawn_status="MOUTH CLOSED"
            cv2.putText(frame,f"EAR: {ear:.3f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame,eye_status,(20,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255) if eye_status=="DROWSY" else (0,255,0),2)
            cv2.putText(frame,f"MAR: {mar:.3f}",(20,115),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
            cv2.putText(frame,yawn_status,(20,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255) if yawn_status=="YAWNING" else (0,255,0),2)
        cv2.imshow("DMS - Eye and Yawning Detection",frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
cap.release()
cv2.destroyAllWindows()