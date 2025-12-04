import cv2
import mediapipe as mp
import numpy as np

# 1. 미디어파이프(MediaPipe) 얼굴 인식 도구 준비
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. 웹캠 켜기 (0번이 기본 카메라)
cap = cv2.VideoCapture(0)

print("프로그램이 시작되었습니다! 'q' 키를 누르면 종료됩니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # 성능을 위해 이미지를 잠시 읽기 전용으로 변경
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. AI에게 얼굴 찾으라고 시키기
    results = face_mesh.process(image)

    # 다시 그리기 모드로 변경
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 4. 눈 좌표 찾기 (랜드마크 인덱스 활용)
            h, w, c = image.shape
            
            # 왼쪽 눈 중심 (대략적인 좌표)
            left_eye = face_landmarks.landmark[33]
            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            
            # 오른쪽 눈 중심
            right_eye = face_landmarks.landmark[263]
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)
            
            # 5. 가상 선글라스 그리기 (OpenCV 기능)
            # 검은색 알 (원)
            cv2.circle(image, (lx + 10, ly), 35, (0, 0, 0), -1) # 왼쪽 알
            cv2.circle(image, (rx - 10, ry), 35, (0, 0, 0), -1) # 오른쪽 알
            
            # 안경 테 (선)
            cv2.line(image, (lx + 10, ly), (rx - 10, ry), (50, 50, 50), 5) 

            # 설명 텍스트
            cv2.putText(image, "Open Source SW Project", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 6. 화면에 보여주기
    cv2.imshow('My AR Filter', image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()