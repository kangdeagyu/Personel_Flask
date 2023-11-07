import joblib
import cv2
import dlib
import numpy as np
from flask import Flask

class AI_Service:
    def AI_predict(image_file):
        try:
            # 이미지 파일을 바이트 스트림으로 읽기
            img_stream = np.fromstring(image_file, np.uint8)

            # OpenCV를 사용하여 이미지로 디코딩
            image = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)

            # 얼굴 검출기 초기화
            face_detector = dlib.get_frontal_face_detector()
            
            # 얼굴 영역 검출
            faces = face_detector(image)

            # 얼굴 랜드마크 검출기 초기화
            landmark_predictor = dlib.shape_predictor('app/static/shape_predictor_68_face_landmarks.dat')

            for face in faces:
                
                # 얼굴 랜드마크 검출
                landmarks = landmark_predictor(image, face)
                
                # 미간 자리 좌표 추출 (27부터 30까지의 인덱스 사용)
                forehead_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 31)])
                
                # 해당 영역의 색상 추출
                forehead_color_bgr = np.mean(image[forehead_points[:, 1], forehead_points[:, 0]], axis=0)
                
                # BGR에서 RGB로 변환
                forehead_color_rgb = list(reversed(forehead_color_bgr))
                
            r = forehead_color_rgb[0]
            g = forehead_color_rgb[1]
            b = forehead_color_rgb[2]

            clf = joblib.load("app/static/voting_clf.h5")
            pre = clf.predict([[r,g,b]])
            return (pre,r,g,b)
        except Exception as e:
            print('error:',e)
            err = ['인식오류']
            return err
