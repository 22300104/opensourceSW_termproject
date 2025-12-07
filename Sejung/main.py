import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# --- [1. 설정 및 초기화] ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- [2. 핵심 함수: 투명 이미지 합성 (Alpha Blending)] ---
def overlay_transparent(background, overlay, x, y):
    try:
        bg_h, bg_w, _ = background.shape
        h, w, _ = overlay.shape

        # 화면 밖으로 나가는 경우 예외 처리 (좌표 보정)
        if x < 0: 
            overlay = overlay[:, -x:]
            w = overlay.shape[1]
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            h = overlay.shape[0]
            y = 0
        if x + w > bg_w:
            overlay = overlay[:, :bg_w - x]
            w = overlay.shape[1]
        if y + h > bg_h:
            overlay = overlay[:bg_h - y, :]
            h = overlay.shape[0]

        # 알파 채널(투명도) 분리 (0~1 사이 값으로 변환)
        alpha = overlay[:, :, 3] / 255.0
        colors = overlay[:, :, :3]
        
        # 합성 공식: (배경 * (1-알파)) + (덮어쓸 이미지 * 알파)
        # 배경 이미지의 해당 영역(ROI)을 가져와서 합성
        roi = background[y:y+h, x:x+w]
        
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1.0 - alpha) + colors[:, :, c] * alpha
            
        background[y:y+h, x:x+w] = roi
        return background
    except Exception as e:
        # 에러 발생 시 그냥 원본 반환 (프로그램 꺼짐 방지)
        return background

# --- [3. 이미지 파일 로드 (경로 문제 해결)] ---
file_name = 'glasses.png'
current_dir = os.getcwd() # 현재 터미널 위치
possible_paths = [
    file_name,                        # 1. 현재 폴더
    os.path.join('Sejung', file_name), # 2. Sejung 폴더 안 (상위에서 실행 시)
    os.path.join('..', file_name)      # 3. 상위 폴더 (하위에서 실행 시)
]

glasses_img = None
for path in possible_paths:
    if os.path.exists(path):
        glasses_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(f"✅ 이미지 로드 성공! 경로: {path}")
        break

if glasses_img is None:
    print(f"❌ 오류: '{file_name}' 파일을 찾을 수 없습니다.")
    print(f"   현재 위치: {current_dir}")
    print("   -> glasses.png 파일이 올바른 폴더(Sejung)에 있는지 확인하세요.")
    # 임시 방편: 빈 이미지 생성 (에러 방지)
    glasses_img = np.zeros((100, 300, 4), dtype=np.uint8)
    cv2.circle(glasses_img, (75, 50), 45, (0, 0, 0, 255), -1)
    cv2.circle(glasses_img, (225, 50), 45, (0, 0, 0, 255), -1)
    print("⚠️ 임시 안경 이미지를 생성하여 실행합니다.")

elif glasses_img.shape[2] < 4:
    print("⚠️ 경고: 이미지에 투명도(Alpha) 채널이 없습니다! 합성이 이상할 수 있습니다.")


# --- [4. 메인 실행 루프] ---
cap = cv2.VideoCapture(0)

print("\n=== AR Face Filter Started ===")
print("1. [기본] 눈을 인식하여 선글라스 착용")
print("2. [이벤트] 입을 크게 벌리면 'Wow!' 효과")
print("종료하려면 'q'를 누르세요.\n")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        break

    # 성능 최적화: 이미지 쓰기 금지 후 처리
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # 그리기 위해 다시 쓰기 허용
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, c = image.shape
            
            # --- [좌표 계산] ---
            # 왼쪽 눈(33)과 오른쪽 눈(263) 좌표
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)
            
            # 입 윗입술(13)과 아랫입술(14)
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]
            lip_dist = int(abs(top_lip.y - bottom_lip.y) * h)

            # --- [기능 1: 선글라스 씌우기] ---
            # 각도 계산 (고개를 기울여도 따라가게)
            dy = ry - ly
            dx = rx - lx
            angle = np.degrees(np.arctan2(dy, dx)) # 라디안 -> 도 변환
            
            # 거리 기반 크기 조절
            eye_dist = np.sqrt(dx**2 + dy**2)
            glass_width = int(eye_dist * 2.3) # 눈 사이 거리의 2.3배 크기
            
            if glass_width > 0:
                # 원본 비율 유지하며 높이 계산
                scale_factor = glass_width / glasses_img.shape[1]
                glass_height = int(glasses_img.shape[0] * scale_factor)
                
                # 이미지 크기 조절
                resized_glass = cv2.resize(glasses_img, (glass_width, glass_height))
                
                # 이미지 회전
                M = cv2.getRotationMatrix2D((glass_width//2, glass_height//2), -angle, 1)
                rotated_glass = cv2.warpAffine(resized_glass, M, (glass_width, glass_height))
                
                # 위치 잡기 (두 눈의 중점)
                center_x = (lx + rx) // 2 - glass_width // 2
                center_y = (ly + ry) // 2 - glass_height // 2
                
                # 합성 함수 호출
                image = overlay_transparent(image, rotated_glass, center_x, center_y)

            # --- [기능 2: 입 벌림 감지] ---
            if lip_dist > 40: # 입이 일정 이상 벌어지면 (값 조절 가능)
                cv2.putText(image, "Wow!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                # 얼굴 주변에 박스 표시
                face_top = int(face_landmarks.landmark[10].y * h)
                face_bot = int(face_landmarks.landmark[152].y * h)
                cv2.rectangle(image, (lx-50, face_top-50), (rx+50, face_bot+50), (0, 255, 255), 3)

    # 화면 출력
    cv2.imshow('AR Filter Project - Sejoong', image)

    # 종료 키
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()