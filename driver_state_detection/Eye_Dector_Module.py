import cv2
import numpy as np
from numpy import linalg as LA
from Utils import resize


class EyeDetector:

    def __init__(self, show_processing: bool = False):
      
        self.keypoints = None
        self.frame = None
        self.show_processing = show_processing
        self.eye_width = None

    def show_eye_keypoints(self, color_frame, landmarks):
     
    
        self.keypoints = landmarks

        for n in range(36, 48):
            x = self.keypoints.part(n).x
            y = self.keypoints.part(n).y
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return

    def get_EAR(self, frame, landmarks):
     

        self.keypoints = landmarks
        self.frame = frame
        pts = self.keypoints

        i = 0   
         
        eye_pts_l = np.zeros(shape=(6, 2))
         
        eye_pts_r = np.zeros(shape=(6, 2))

        for n in range(36, 42):   
            point_l = pts.part(n)   
            point_r = pts.part(n + 6)   
             
            eye_pts_l[i] = [point_l.x, point_l.y]
            
            eye_pts_r[i] = [point_r.x, point_r.y]
            i += 1   
        def EAR_eye(eye_pts):
            
            ear_eye = (LA.norm(eye_pts[1] - eye_pts[5]) + LA.norm(
                eye_pts[2] - eye_pts[4])) / (2 * LA.norm(eye_pts[0] - eye_pts[3]))
           
            return ear_eye

        ear_left = EAR_eye(eye_pts_l)   
        ear_right = EAR_eye(eye_pts_r)   
         
        ear_avg = (ear_left + ear_right) / 2

        return ear_avg

    def get_Gaze_Score(self, frame, landmarks):
        
        self.keypoints = landmarks
        self.frame = frame

        def get_ROI(left_corner_keypoint_num: int):
           

            kp_num = left_corner_keypoint_num

            eye_array = np.array(
                [(self.keypoints.part(kp_num).x, self.keypoints.part(kp_num).y),
                 (self.keypoints.part(kp_num+1).x,
                  self.keypoints.part(kp_num+1).y),
                 (self.keypoints.part(kp_num+2).x,
                  self.keypoints.part(kp_num+2).y),
                 (self.keypoints.part(kp_num+3).x,
                  self.keypoints.part(kp_num+3).y),
                 (self.keypoints.part(kp_num+4).x,
                  self.keypoints.part(kp_num+4).y),
                 (self.keypoints.part(kp_num+5).x, self.keypoints.part(kp_num+5).y)], np.int32)

            min_x = np.min(eye_array[:, 0])
            max_x = np.max(eye_array[:, 0])
            min_y = np.min(eye_array[:, 1])
            max_y = np.max(eye_array[:, 1])

            eye_roi = self.frame[min_y-2:max_y+2, min_x-2:max_x+2]

            return eye_roi

        def get_gaze(eye_roi):
           

            eye_center = np.array(
                [(eye_roi.shape[1] // 2), (eye_roi.shape[0] // 2)])   
            gaze_score = None
            circles = None

             
            eye_roi = cv2.bilateralFilter(eye_roi, 4, 40, 40)

            circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1, 10,
                                       param1=90, param2=6, minRadius=1, maxRadius=9)
            

            if circles is not None and len(circles) > 0:
                circles = np.uint16(np.around(circles))
                circle = circles[0][0, :]

                cv2.circle(
                    eye_roi, (circle[0], circle[1]), circle[2], (255, 255, 255), 1)
                cv2.circle(
                    eye_roi, (circle[0], circle[1]), 1, (255, 255, 255), -1)

                 
                pupil_position = np.array([int(circle[0]), int(circle[1])])

                cv2.line(eye_roi, (eye_center[0], eye_center[1]), (
                    pupil_position[0], pupil_position[1]), (255, 255, 255), 1)

                gaze_score = LA.norm(
                    pupil_position - eye_center) / eye_center[0]
                 

            cv2.circle(eye_roi, (eye_center[0],
                                 eye_center[1]), 1, (0, 0, 0), -1)

            if gaze_score is not None:
                return gaze_score, eye_roi
            else:
                return None, None

        left_eye_ROI = get_ROI(36)   
        right_eye_ROI = get_ROI(42)   

         
        gaze_eye_left, left_eye = get_gaze(left_eye_ROI)
        gaze_eye_right, right_eye = get_gaze(right_eye_ROI)

         
        if self.show_processing and (left_eye is not None) and (right_eye is not None):
            left_eye = resize(left_eye, 1000)
            right_eye = resize(right_eye, 1000)
            cv2.imshow("left eye", left_eye)
            cv2.imshow("right eye", right_eye)

        if gaze_eye_left and gaze_eye_right:

             
            avg_gaze_score = (gaze_eye_left + gaze_eye_left) / 2
            return avg_gaze_score

        else:
            return None
