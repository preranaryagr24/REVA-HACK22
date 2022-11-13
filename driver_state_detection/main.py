import time
from pygame import mixer
mixer.init()
mixer.music.load('m.mp3')


 

import cv2
import dlib
import numpy as np

from Utils import get_face_area
from Eye_Dector_Module import EyeDetector as EyeDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from Attention_Scorer_Module import AttentionScorer as AttScorer

 
CAPTURE_SOURCE = 0

 
camera_matrix = np.array(
    [[899.12150372, 0., 644.26261492],
     [0., 899.45280671, 372.28009436],
     [0, 0,  1]], dtype="double")

 
dist_coeffs = np.array(
    [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double")


def main():

    ctime = 0   
    ptime = 0   
    prev_time = 0   
    fps_lim = 11   
    time_lim = 1. / fps_lim   

    cv2.setUseOptimized(True)   

    
    Detector = dlib.get_frontal_face_detector()
    Predictor = dlib.shape_predictor(
        "predictor/shape_predictor_68_face_landmarks.dat")   
    

     
    Eye_det = EyeDet(show_processing=False)

    Head_pose = HeadPoseEst(show_axis=True)

    
    
    Scorer = AttScorer(fps_lim, ear_tresh=0.15, ear_time_tresh=2, gaze_tresh=0.2,
                       gaze_time_tresh=2, pitch_tresh=35, yaw_tresh=28, pose_time_tresh=2.5, verbose=False)
    
    
    cap = cv2.VideoCapture(CAPTURE_SOURCE)
    if not cap.isOpened():   
        print("Cannot open camera")
        exit()

    while True:   

        delta_time = time.perf_counter() - prev_time   
        ret, frame = cap.read()   

        if not ret:   
            print("Can't receive frame from camera/stream end")
            break

          
        if CAPTURE_SOURCE == 0:
            frame = cv2.flip(frame, 2)

        if delta_time >= time_lim:   
            prev_time = time.perf_counter()

             
            ctime = time.perf_counter()
            fps = 1.0 / float(ctime - ptime)
            ptime = ctime
            cv2.putText(frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)

             
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             
            gray = cv2.bilateralFilter(gray, 5, 10, 10)

             
            faces = Detector(gray)

            if len(faces) > 0:   

                 
                faces = sorted(faces, key=get_face_area, reverse=True)
                driver_face = faces[0]

                 
                landmarks = Predictor(gray, driver_face)

                 
                Eye_det.show_eye_keypoints(
                    color_frame=frame, landmarks=landmarks)

                 
                ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

                 
                tired, perclos_score = Scorer.get_PERCLOS(ear)

                 
                gaze = Eye_det.get_Gaze_Score(
                    frame=gray, landmarks=landmarks)

                 
                frame_det, roll, pitch, yaw = Head_pose.get_pose(
                    frame=frame, landmarks=landmarks)

                 
                if frame_det is not None:
                    frame = frame_det

                 
                if ear is not None:
                    cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                 
                if gaze is not None:
                    cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                 
                cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                
                 
                if tired:  
                    cv2.putText(frame, "TIRED!", (10, 280),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                
                asleep, looking_away, distracted = Scorer.eval_scores(
                    ear, gaze, roll, pitch, yaw)  

                 
                if asleep:
                    cv2.putText(frame, "ASLEEP!", (10, 300),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    mixer.music.play()
                    time.sleep(1)
                    mixer.music.stop()

                if looking_away:
                    cv2.putText(frame, "LOOKING AWAY!", (10, 320),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    mixer.music.play()
                    time.sleep(1)
                    mixer.music.stop()
                if distracted:
                    cv2.putText(frame, "DISTRACTED!", (10, 340),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    mixer.music.play()
                    time.sleep(1)
                    mixer.music.stop()

            cv2.imshow("Frame", frame)  # show the frame on screen

         
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
