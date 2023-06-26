import cv2
import mediapipe as mp
import numpy as np

#Getting mediapipe Pose ready
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a) # Start point
    b = np.array(b) # Mid point
    c = np.array(c) # End point
        
    ba = a-b
    bc = c-b
    
    cosine_angle = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    
    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle2d = np.abs(radians*180/np.pi)
    
    if angle2d >180.0:
        angle2d = 360-angle2d
    
    return angle

# Will determine the distance between two points given two vector inputs
def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    
    distance = np.linalg.norm(b-a)
    return distance

def controller():
    #Capture webcam
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    control = 'default'
    #Initialize the body segments
    #Get coordinates of body landmarks and save them as more logical values
    #HEAD
    head_points = [0,0,0]
            
    #UPPER BODY
    left_shoulder = [0,0,0]
    left_elbow = [0,0,0]
    left_wrist = [0,0,0]

    right_shoulder = [0,0,0]
    right_elbow = [0,0,0]
    right_wrist = [0,0,0]
            
    #LOWER BODY
    left_hip = [0.,0.,0.]
    left_knee = [0.,0.,0.]
    left_ankle = [0.,0.,0.]
    left_heel = [0.,0.,0.]            
    left_toe = [0.,0.,0.]
    right_hip = [0.,0.,0.]
    right_knee = [0.,0.,0.]
    right_ankle = [0.,0.,0.]
    right_heel = [0.,0.,0.]
    right_toe = [0.,0.,0.]

    #Initialize relevant angles
    right_shoulder_angle = 0.
    left_shoulder_angle = 0.
    right_elbow_angle = 0.
    left_elbow_angle = 0.
    right_knee_angle = 0.
    left_knee_angle = 0.
    right_hip_angle = 0.
    left_hip_angle = 0.
    right_ankle_angle = 0.
    left_ankle_angle = 0.           

    #THE MAIN FUNCTION THAT RUNS THE GAME
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cam.isOpened():
            success, frame = cam.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image = cv2.flip(image,1)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                #Get coordinates of body landmarks and save them as more logical values
                #HEAD
                head_points = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y, landmarks[mp_pose.PoseLandmark.NOSE.value].z]
                #UPPER BODY
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                #LOWER BODY
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]            
                left_toe = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]
                right_toe = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
                #Calculate relevant angles
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)
                left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                right_ankle_angle = calculate_angle(right_knee, right_heel, right_toe)
                left_ankle_angle = calculate_angle(left_knee, left_heel, left_toe) 
                
                #Shoulder Flexion
                if (right_shoulder_angle > 150) and (right_shoulder_angle < 180):
                    control = 'up'
                elif (left_shoulder_angle > 150) and (left_shoulder_angle < 180):
                    control = 'up'
                #Shoulder Adduction
                elif (right_shoulder_angle > 80) and (right_shoulder_angle < 100):
                    control = 'right'
                elif (left_shoulder_angle > 80) and (left_shoulder_angle < 100):
                    control = 'left'
                #Squat
                elif (left_knee_angle < 90.0) and (right_knee_angle < 90):
                    control = 'down'
                else:
                    control = 'default'
            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.namedWindow("BALANCE CAM")
            cv2.moveWindow("BALANCE CAM", 0, 121)
            cv2.imshow("BALANCE CAM", image)
            cv2.waitKey(10)
            if cv2.waitKey(10) == 27: # This puts you out of the loop above if you hit escape
                cam.release() # Releases the webcam from your memory
                cv2.destroyAllWindows() # Closes the window for the webcam
            yield control


for control in controller():
    print(control)