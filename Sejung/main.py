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

# --- [2. 필터 생성 함수들] ---
def create_glasses_filter(width, height):
    """안경 필터 생성"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    # 두 개의 원형 렌즈
    lens_radius = min(width, height) // 4
    cv2.circle(img, (width//4, height//2), lens_radius, (50, 50, 50, 200), -1)
    cv2.circle(img, (3*width//4, height//2), lens_radius, (50, 50, 50, 200), -1)
    # 프레임
    cv2.circle(img, (width//4, height//2), lens_radius, (0, 0, 0, 255), 3)
    cv2.circle(img, (3*width//4, height//2), lens_radius, (0, 0, 0, 255), 3)
    # 다리
    cv2.line(img, (width//4 - lens_radius, height//2), (0, height//2), (0, 0, 0, 255), 3)
    cv2.line(img, (3*width//4 + lens_radius, height//2), (width, height//2), (0, 0, 0, 255), 3)
    # 다리 연결
    cv2.line(img, (width//4, height//2 - lens_radius//2), (3*width//4, height//2 - lens_radius//2), (0, 0, 0, 255), 3)
    return img

def create_hat_filter(width, height):
    """모자 필터 생성"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    # 모자 본체 (타원형)
    cv2.ellipse(img, (width//2, height//3), (width//2, height//3), 0, 0, 180, (139, 69, 19, 255), -1)
    # 모자 테두리
    cv2.ellipse(img, (width//2, height//3), (width//2, height//3), 0, 0, 180, (0, 0, 0, 255), 3)
    # 모자 장식 (리본)
    cv2.rectangle(img, (width//2 - 20, height//3 - 5), (width//2 + 20, height//3 + 5), (255, 0, 0, 255), -1)
    return img

def create_mustache_filter(width, height):
    """수염 필터 생성"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    # 수염 본체 (타원형)
    cv2.ellipse(img, (width//2, height//2), (width//3, height//4), 0, 0, 360, (50, 50, 50, 220), -1)
    # 수염 테두리
    cv2.ellipse(img, (width//2, height//2), (width//3, height//4), 0, 0, 360, (0, 0, 0, 255), 2)
    # 양쪽 끝 강조
    cv2.ellipse(img, (width//4, height//2), (width//8, height//6), 0, 0, 360, (50, 50, 50, 220), -1)
    cv2.ellipse(img, (3*width//4, height//2), (width//8, height//6), 0, 0, 360, (50, 50, 50, 220), -1)
    return img

def create_crown_filter(width, height):
    """왕관 필터 생성"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    # 왕관 본체
    points = np.array([
        [width//2, 0],
        [width//2 - width//4, height//2],
        [width//4, height//2],
        [width//2, height//3],
        [3*width//4, height//2],
        [width//2 + width//4, height//2]
    ], np.int32)
    cv2.fillPoly(img, [points], (255, 215, 0, 255))
    cv2.polylines(img, [points], True, (0, 0, 0, 255), 2)
    # 보석 장식
    cv2.circle(img, (width//2, height//6), 5, (255, 0, 0, 255), -1)
    cv2.circle(img, (width//4, height//3), 4, (0, 255, 0, 255), -1)
    cv2.circle(img, (3*width//4, height//3), 4, (0, 0, 255, 255), -1)
    return img

# --- [3. 핵심 함수: 투명 이미지 합성 (Alpha Blending)] ---
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

# --- [4. 이미지 파일 로드 (경로 문제 해결)] ---
file_name = 'glasses.png'
current_dir = os.getcwd()
possible_paths = [
    file_name,
    os.path.join('Sejung', file_name),
    os.path.join('..', file_name)
]

glasses_img = None
for path in possible_paths:
    if os.path.exists(path):
        glasses_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(f"✅ 이미지 로드 성공! 경로: {path}")
        break

if glasses_img is None:
    print(f"⚠️ '{file_name}' 파일을 찾을 수 없어 코드로 생성된 필터를 사용합니다.")
    glasses_img = create_glasses_filter(300, 100)
elif glasses_img.shape[2] < 4:
    print("⚠️ 경고: 이미지에 투명도(Alpha) 채널이 없습니다! 합성이 이상할 수 있습니다.")
    # 알파 채널 추가
    glasses_img = cv2.cvtColor(glasses_img, cv2.COLOR_BGR2BGRA)

# --- [5. 필터 관리 시스템] ---
current_filter = 'glasses'  # 기본 필터
filter_names = {
    'glasses': '안경',
    'hat': '모자',
    'mustache': '수염',
    'crown': '왕관',
    'none': '없음'
}


# --- [6. 필터 적용 함수] ---
def apply_filter(image, face_landmarks, filter_type, h, w):
    """얼굴 랜드마크에 따라 필터를 적용하는 함수"""
    if filter_type == 'none':
        return image
    
    # 공통 좌표 계산
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)
    
    dx = rx - lx
    dy = ry - ly
    angle = np.degrees(np.arctan2(dy, dx))
    eye_dist = np.sqrt(dx**2 + dy**2)
    
    if filter_type == 'glasses':
        # 안경 필터
        glass_width = int(eye_dist * 2.3)
        if glass_width > 0:
            glass_height = int(glass_width * 0.4)
            filter_img = create_glasses_filter(glass_width, glass_height) if glasses_img is None else glasses_img.copy()
            
            if filter_img.shape[1] != glass_width:
                scale_factor = glass_width / filter_img.shape[1]
                glass_height = int(filter_img.shape[0] * scale_factor)
                filter_img = cv2.resize(filter_img, (glass_width, glass_height))
            
            # 회전
            M = cv2.getRotationMatrix2D((glass_width//2, glass_height//2), -angle, 1)
            rotated_filter = cv2.warpAffine(filter_img, M, (glass_width, glass_height))
            
            center_x = (lx + rx) // 2 - glass_width // 2
            center_y = (ly + ry) // 2 - glass_height // 2
            image = overlay_transparent(image, rotated_filter, center_x, center_y)
    
    elif filter_type == 'hat':
        # 모자 필터 (이마 위)
        forehead = face_landmarks.landmark[10]
        fx, fy = int(forehead.x * w), int(forehead.y * h)
        hat_width = int(eye_dist * 2.5)
        hat_height = int(hat_width * 0.6)
        
        filter_img = create_hat_filter(hat_width, hat_height)
        M = cv2.getRotationMatrix2D((hat_width//2, hat_height//2), -angle, 1)
        rotated_filter = cv2.warpAffine(filter_img, M, (hat_width, hat_height))
        
        center_x = fx - hat_width // 2
        center_y = fy - hat_height - 20
        image = overlay_transparent(image, rotated_filter, center_x, center_y)
    
    elif filter_type == 'mustache':
        # 수염 필터 (코 아래)
        nose_tip = face_landmarks.landmark[4]
        upper_lip = face_landmarks.landmark[13]
        nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)
        ux, uy = int(upper_lip.x * w), int(upper_lip.y * h)
        
        mustache_width = int(eye_dist * 1.8)
        mustache_height = int(mustache_width * 0.5)
        
        filter_img = create_mustache_filter(mustache_width, mustache_height)
        M = cv2.getRotationMatrix2D((mustache_width//2, mustache_height//2), -angle, 1)
        rotated_filter = cv2.warpAffine(filter_img, M, (mustache_width, mustache_height))
        
        center_x = (nx + ux) // 2 - mustache_width // 2
        center_y = (ny + uy) // 2
        image = overlay_transparent(image, rotated_filter, center_x, center_y)
    
    elif filter_type == 'crown':
        # 왕관 필터 (머리 위)
        forehead = face_landmarks.landmark[10]
        fx, fy = int(forehead.x * w), int(forehead.y * h)
        crown_width = int(eye_dist * 2.2)
        crown_height = int(crown_width * 0.8)
        
        filter_img = create_crown_filter(crown_width, crown_height)
        M = cv2.getRotationMatrix2D((crown_width//2, crown_height//2), -angle, 1)
        rotated_filter = cv2.warpAffine(filter_img, M, (crown_width, crown_height))
        
        center_x = fx - crown_width // 2
        center_y = fy - crown_height - 30
        image = overlay_transparent(image, rotated_filter, center_x, center_y)
    
    return image

# --- [7. 메인 실행 루프] ---
cap = cv2.VideoCapture(0)

print("\n=== AR Face Filter Started ===")
print("🎭 필터 전환: 숫자 키를 누르세요")
print("   [1] 안경  [2] 모자  [3] 수염  [4] 왕관  [0] 없음")
print("📸 기능: 입을 크게 벌리면 'Wow!' 효과")
print("❌ 종료: 'q' 키를 누르세요\n")

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
            
            # --- [필터 적용] ---
            image = apply_filter(image, face_landmarks, current_filter, h, w)
            
            # --- [입 벌림 감지] ---
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]
            lip_dist = int(abs(top_lip.y - bottom_lip.y) * h)
            
            if lip_dist > 40:
                cv2.putText(image, "Wow!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                # 얼굴 주변에 박스 표시
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                lx, ly = int(left_eye.x * w), int(left_eye.y * h)
                rx, ry = int(right_eye.x * w), int(right_eye.y * h)
                face_top = int(face_landmarks.landmark[10].y * h)
                face_bot = int(face_landmarks.landmark[152].y * h)
                cv2.rectangle(image, (lx-50, face_top-50), (rx+50, face_bot+50), (0, 255, 255), 3)
    
    # --- [화면에 현재 필터 표시] ---
    filter_text = f"현재 필터: {filter_names[current_filter]}"
    cv2.putText(image, filter_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, "[1]안경 [2]모자 [3]수염 [4]왕관 [0]없음", (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 화면 출력
    cv2.imshow('AR Filter Project - Sejoong', image)

    # 키보드 입력 처리
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        current_filter = 'glasses'
        print(f"✅ 필터 변경: {filter_names[current_filter]}")
    elif key == ord('2'):
        current_filter = 'hat'
        print(f"✅ 필터 변경: {filter_names[current_filter]}")
    elif key == ord('3'):
        current_filter = 'mustache'
        print(f"✅ 필터 변경: {filter_names[current_filter]}")
    elif key == ord('4'):
        current_filter = 'crown'
        print(f"✅ 필터 변경: {filter_names[current_filter]}")
    elif key == ord('0'):
        current_filter = 'none'
        print(f"✅ 필터 변경: {filter_names[current_filter]}")

cap.release()
cv2.destroyAllWindows()