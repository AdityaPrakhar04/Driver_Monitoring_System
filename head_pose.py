import cv2
import mediapipe as mp
import numpy as np
BaseOptions=mp.tasks.BaseOptions
FaceLandmarker=mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions=mp.tasks.vision.FaceLandmarkerOptions
RunningMode=mp.tasks.vision.RunningMode
options=FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path="face_landmarker.task"),running_mode=RunningMode.IMAGE,num_faces=1)
cap=cv2.VideoCapture(0)
NOSE=1
CHIN=152
LEFT_EYE=33
RIGHT_EYE=263
LEFT_MOUTH=61
RIGHT_MOUTH=291
def get_head_pose(landmarks,frame):
    h,w,_=frame.shape
    image_points=np.array([(landmarks[NOSE].x*w,landmarks[NOSE].y*h),(landmarks[CHIN].x*w,landmarks[CHIN].y*h),(landmarks[LEFT_EYE].x*w,landmarks[LEFT_EYE].y*h),(landmarks[RIGHT_EYE].x*w,landmarks[RIGHT_EYE].y*h),(landmarks[LEFT_MOUTH].x*w,landmarks[LEFT_MOUTH].y*h),(landmarks[RIGHT_MOUTH].x*w,landmarks[RIGHT_MOUTH].y*h)],dtype=np.float64)
    model_points=np.array([(0.0,0.0,0.0),(0.0,-63.6,-12.5),(-43.3,32.7,-26.0),(43.3,32.7,-26.0),(-28.9,-28.9,-24.1),(28.9,-28.9,-24.1)])
    focal_length=w
    center=(w/2,h/2)
    camera_matrix=np.array([[focal_length,0,center[0]],[0,focal_length,center[1]],[0,0,1]],dtype=np.float64)
    dist_coeffs=np.zeros((4,1))
    success,rotation_vector,translation_vector=cv2.solvePnP(model_points,image_points,camera_matrix,dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None
    rotation_matrix,_=cv2.Rodrigues(rotation_vector)
    angles=cv2.RQDecomp3x3(rotation_matrix)[0]
    pitch=angles[0]
    yaw=angles[1]
    roll=angles[2]
    return pitch,yaw,roll
def get_head_status(pitch,yaw):
    if abs(yaw)>20:
        return "DISTRACTED - LEFT/RIGHT"
    elif abs(pitch)>20:
        return "DISTRACTED - UP/DOWN"
    else:
        return "LOOKING FORWARD"
with FaceLandmarker.create_from_options(options) as landmarker:
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
            pose=get_head_pose(landmarks,frame)
            if pose is not None:
                pitch,yaw,roll=pose
                head_status=get_head_status(pitch,yaw)
                cv2.putText(frame,f"Pitch: {pitch:.1f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                cv2.putText(frame,f"Yaw: {yaw:.1f}",(20,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                cv2.putText(frame,f"Roll: {roll:.1f}",(20,110),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                if "DISTRACTED" in head_status:
                    status_color=(0,0,255)
                else:
                    status_color=(0,255,0)
                cv2.putText(frame,head_status,(20,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,status_color,2)
        cv2.imshow("DMS - Head Pose Detection",frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
cap.release()
cv2.destroyAllWindows()