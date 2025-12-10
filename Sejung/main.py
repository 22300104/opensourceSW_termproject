import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

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

# --- [4. 이미지 파일 로드 함수] ---
def load_filter_image(filename, default_width=300, default_height=100, create_func=None):
    """필터 이미지를 로드하는 함수"""
    current_dir = os.getcwd()
    possible_paths = [
        filename,
        os.path.join('Sejung', filename),
        os.path.join('..', filename)
    ]
    
    img = None
    for path in possible_paths:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"✅ 이미지 로드 성공! 경로: {path}")
                break
    
    if img is None:
        if create_func:
            print(f"⚠️ '{filename}' 파일을 찾을 수 없어 코드로 생성된 필터를 사용합니다.")
            img = create_func(default_width, default_height)
        else:
            print(f"⚠️ '{filename}' 파일을 찾을 수 없습니다.")
            return None
    elif img.shape[2] < 4:
        print(f"⚠️ 경고: '{filename}'에 투명도(Alpha) 채널이 없습니다! 알파 채널을 추가합니다.")
        # 알파 채널 추가
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    return img

# 모든 필터 이미지 로드
print("\n=== 필터 이미지 로드 중 ===")
glasses_img = load_filter_image('glasses.png', 300, 100, create_glasses_filter)
hat_img = load_filter_image('hat.png', 300, 180, create_hat_filter)
mustache_img = load_filter_image('mustache.png', 300, 150, create_mustache_filter)
crown_img = load_filter_image('crown.png', 300, 240, create_crown_filter)
print("=" * 30 + "\n")

# --- [5. 한글 텍스트 출력 함수] ---
def put_korean_text(img, text, position, font_size=30, color=(0, 255, 0)):
    """한글 텍스트를 이미지에 출력하는 함수"""
    try:
        # PIL로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 폰트 로드 시도 (여러 경로 시도)
        font = None
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",        # 굴림
            "C:/Windows/Fonts/batang.ttc",       # 바탕
            "malgun.ttf",
            "gulim.ttc",
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            except:
                continue
        
        if font is None:
            try:
                font = ImageFont.load_default()
            except:
                pass
        
        # 텍스트 그리기
        if font:
            draw.text(position, text, font=font, fill=color)
        else:
            draw.text(position, text, fill=color)
        
        # OpenCV 형식으로 다시 변환
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # 에러 발생 시 영문으로 대체
        try:
            cv2.putText(img, text.encode('ascii', 'ignore').decode('ascii'), position, 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size/30, color, 2)
        except:
            pass
    return img

# --- [6. 필터 위치 조정 파라미터 (여기서 수정하세요!)] ---
# 각 필터의 크기 비율과 위치 오프셋을 조정할 수 있습니다
# 프로그램 실행 후 필터 위치가 맞지 않으면 이 값들을 조정하세요!
FILTER_SETTINGS = {
    'glasses': {
        'size_ratio': 2.3,      # 눈 사이 거리의 몇 배로 할지 (크기 조절)
        'height_ratio': 0.4,     # 너비 대비 높이 비율 (실제 이미지 사용 시 무시됨)
        'offset_x': 0,          # X축 오프셋 (양수: 오른쪽, 음수: 왼쪽)
        'offset_y': 0,          # Y축 오프셋 (양수: 아래로, 음수: 위로)
    },
    'hat': {
        'size_ratio': 2.5,      # 크기 조절 (값을 크게 하면 모자가 커짐)
        'height_ratio': 0.6,     # 높이 비율
        'offset_x': 0,          # 좌우 이동 (양수: 오른쪽, 음수: 왼쪽)
        'offset_y': 60,         # 상하 이동 (양수: 아래로, 음수: 위로) - 모자는 낮게 (가장 큰 움직임)
    },
    'mustache': {
        'size_ratio': 1.8,      # 크기 조절
        'height_ratio': 0.5,     # 높이 비율
        'offset_x': 0,          # 좌우 이동
        'offset_y': -25,        # 상하 이동 (양수: 아래, 음수: 위로) - 수염은 높게 (중간 움직임)
    },
    'crown': {
        'size_ratio': 2.2,      # 크기 조절
        'height_ratio': 0.8,     # 높이 비율
        'offset_x': 0,          # 좌우 이동
        'offset_y': 15,         # 상하 이동 (양수: 아래로, 음수: 위로) - 왕관은 낮게 (가장 작은 움직임)
    }
}

# --- [6-1. 전역 조절 파라미터] ---
SIZE_SCALE = 1.0          # 필터 크기 배율 (실시간 조절)
ALPHA_SCALE = 1.0         # 필터 투명도 배율 (실시간 조절)
SIZE_STEP = 0.1
ALPHA_STEP = 0.1
SIZE_MIN, SIZE_MAX = 0.5, 3.0
ALPHA_MIN, ALPHA_MAX = 0.1, 2.0

# --- [7. 필터 관리 시스템] ---
# 여러 필터를 동시에 적용할 수 있도록 리스트로 관리
active_filters = ['glasses']  # 기본 활성 필터 목록
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
    
    # 설정 가져오기
    settings = FILTER_SETTINGS.get(filter_type, {})
    # 전역 크기/투명도 배율 적용
    size_ratio = settings.get('size_ratio', 2.0) * SIZE_SCALE
    height_ratio = settings.get('height_ratio', 0.5)
    offset_x = settings.get('offset_x', 0)
    offset_y = settings.get('offset_y', 0)
    
    if filter_type == 'glasses':
        # 안경 필터
        glass_width = int(eye_dist * size_ratio)
        if glass_width > 0:
            if glasses_img is not None:
                # 실제 이미지 사용
                scale_factor = glass_width / glasses_img.shape[1]
                glass_height = int(glasses_img.shape[0] * scale_factor)
                filter_img = cv2.resize(glasses_img.copy(), (glass_width, glass_height))
            else:
                # 코드로 생성
                glass_height = int(glass_width * height_ratio)
                filter_img = create_glasses_filter(glass_width, glass_height)
            
            # 회전
            M = cv2.getRotationMatrix2D((glass_width//2, glass_height//2), -angle, 1)
            rotated_filter = cv2.warpAffine(filter_img, M, (glass_width, glass_height))
            if rotated_filter.shape[2] == 4 and ALPHA_SCALE != 1.0:
                rotated_filter[:, :, 3] = np.clip(rotated_filter[:, :, 3] * ALPHA_SCALE, 0, 255)
            
            center_x = (lx + rx) // 2 - glass_width // 2 + offset_x
            center_y = (ly + ry) // 2 - glass_height // 2 + offset_y
            image = overlay_transparent(image, rotated_filter, center_x, center_y)
    
    elif filter_type == 'hat':
        # 모자 필터 (이마 위)
        forehead = face_landmarks.landmark[10]
        fx, fy = int(forehead.x * w), int(forehead.y * h)
        hat_width = int(eye_dist * size_ratio)
        
        if hat_img is not None:
            # 실제 이미지 사용
            scale_factor = hat_width / hat_img.shape[1]
            hat_height = int(hat_img.shape[0] * scale_factor)
            filter_img = cv2.resize(hat_img.copy(), (hat_width, hat_height))
        else:
            # 코드로 생성
            hat_height = int(hat_width * height_ratio)
            filter_img = create_hat_filter(hat_width, hat_height)
        
        M = cv2.getRotationMatrix2D((hat_width//2, hat_height//2), -angle, 1)
        rotated_filter = cv2.warpAffine(filter_img, M, (hat_width, hat_height))
        if rotated_filter.shape[2] == 4 and ALPHA_SCALE != 1.0:
            rotated_filter[:, :, 3] = np.clip(rotated_filter[:, :, 3] * ALPHA_SCALE, 0, 255)
        
        center_x = fx - hat_width // 2 + offset_x
        center_y = fy - hat_height + offset_y
        image = overlay_transparent(image, rotated_filter, center_x, center_y)
    
    elif filter_type == 'mustache':
        # 수염 필터 (코 아래)
        nose_tip = face_landmarks.landmark[4]
        upper_lip = face_landmarks.landmark[13]
        nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)
        ux, uy = int(upper_lip.x * w), int(upper_lip.y * h)
        
        mustache_width = int(eye_dist * size_ratio)
        
        if mustache_img is not None:
            # 실제 이미지 사용
            scale_factor = mustache_width / mustache_img.shape[1]
            mustache_height = int(mustache_img.shape[0] * scale_factor)
            filter_img = cv2.resize(mustache_img.copy(), (mustache_width, mustache_height))
        else:
            # 코드로 생성
            mustache_height = int(mustache_width * height_ratio)
            filter_img = create_mustache_filter(mustache_width, mustache_height)
        
        M = cv2.getRotationMatrix2D((mustache_width//2, mustache_height//2), -angle, 1)
        rotated_filter = cv2.warpAffine(filter_img, M, (mustache_width, mustache_height))
        if rotated_filter.shape[2] == 4 and ALPHA_SCALE != 1.0:
            rotated_filter[:, :, 3] = np.clip(rotated_filter[:, :, 3] * ALPHA_SCALE, 0, 255)
        
        center_x = (nx + ux) // 2 - mustache_width // 2 + offset_x
        center_y = (ny + uy) // 2 + offset_y
        image = overlay_transparent(image, rotated_filter, center_x, center_y)
    
    elif filter_type == 'crown':
        # 왕관 필터 (머리 위)
        forehead = face_landmarks.landmark[10]
        fx, fy = int(forehead.x * w), int(forehead.y * h)
        crown_width = int(eye_dist * size_ratio)
        
        if crown_img is not None:
            # 실제 이미지 사용
            scale_factor = crown_width / crown_img.shape[1]
            crown_height = int(crown_img.shape[0] * scale_factor)
            filter_img = cv2.resize(crown_img.copy(), (crown_width, crown_height))
        else:
            # 코드로 생성
            crown_height = int(crown_width * height_ratio)
            filter_img = create_crown_filter(crown_width, crown_height)
        
        M = cv2.getRotationMatrix2D((crown_width//2, crown_height//2), -angle, 1)
        rotated_filter = cv2.warpAffine(filter_img, M, (crown_width, crown_height))
        if rotated_filter.shape[2] == 4 and ALPHA_SCALE != 1.0:
            rotated_filter[:, :, 3] = np.clip(rotated_filter[:, :, 3] * ALPHA_SCALE, 0, 255)
        
        center_x = fx - crown_width // 2 + offset_x
        center_y = fy - crown_height + offset_y
        image = overlay_transparent(image, rotated_filter, center_x, center_y)
    
    return image

# --- [6-1. 다중 필터 적용 함수] ---
def apply_filters(image, face_landmarks, filters, h, w):
    """여러 필터를 순차적으로 적용"""
    if not filters:
        return image
    for f in filters:
        image = apply_filter(image, face_landmarks, f, h, w)
    return image

# --- [7. 스크린샷 저장 함수] ---
def save_screenshot(image, filter_name='none'):
    """현재 화면을 이미지 파일로 저장"""
    try:
        # 저장 폴더 설정
        save_dir = 'screenshots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 파일명 생성 (타임스탬프 + 필터명)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/screenshot_{filter_name}_{timestamp}.jpg"
        
        # 이미지 저장
        cv2.imwrite(filename, image)
        return filename
    except Exception as e:
        return None

# --- [8. 메인 실행 루프] ---
cap = cv2.VideoCapture(0)

# 화면 메시지 관리
status_message = ""
message_timer = 0
MESSAGE_DISPLAY_TIME = 60  # 프레임 수 (약 1초, 60fps 기준)

print("\n=== AR Face Filter Started ===")
print("프로그램이 시작되었습니다.\n")

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

    # 현재 프레임 크기 (얼굴 감지 유무와 관계없이 사용)
    h, w, c = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # --- [필터 적용] ---
            image = apply_filters(image, face_landmarks, active_filters, h, w)
            
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
    if active_filters:
        active_names = [filter_names.get(f, f) for f in active_filters]
        filter_text = f"현재 필터: {', '.join(active_names)}"
    else:
        filter_text = "현재 필터: 없음"
    image = put_korean_text(image, filter_text, (10, 10), font_size=24, color=(0, 255, 0))
    image = put_korean_text(
        image,
        "[1]안경 [2]모자 [3]수염 [4]왕관 [0]모두해제 [+/-]크기 [ [/] ]알파 [s]스크린샷 [q]종료",
        (10, h - 30),
        font_size=18,
        color=(255, 255, 255),
    )
    # 크기/투명도 현재값 표시
    size_alpha_text = f"크기배율: {SIZE_SCALE:.1f} | 알파배율: {ALPHA_SCALE:.1f}"
    image = put_korean_text(image, size_alpha_text, (10, h - 55), font_size=18, color=(0, 200, 255))
    
    # --- [상태 메시지 표시] ---
    if status_message and message_timer > 0:
        # 메시지 표시 (텍스트만)
        image = put_korean_text(image, status_message, (10, h - 60), font_size=20, color=(0, 255, 0))
        message_timer -= 1

    # 화면 출력
    cv2.imshow('AR Filter Project - Sejoong', image)

    # 키보드 입력 처리
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') or key == ord('S'):
        # 스크린샷 저장
        filter_label = "none" if not active_filters else "_".join(active_filters)
        saved_path = save_screenshot(image, filter_label)
        if saved_path:
            status_message = "📸 스크린샷 저장 완료!"
            message_timer = MESSAGE_DISPLAY_TIME
        else:
            status_message = "❌ 스크린샷 저장 실패"
            message_timer = MESSAGE_DISPLAY_TIME
    elif key == ord('1'):
        if 'glasses' in active_filters:
            active_filters.remove('glasses')
        else:
            active_filters.append('glasses')
        status_message = "✅ 필터 토글: 안경"
        message_timer = MESSAGE_DISPLAY_TIME
    elif key == ord('2'):
        if 'hat' in active_filters:
            active_filters.remove('hat')
        else:
            active_filters.append('hat')
        status_message = "✅ 필터 토글: 모자"
        message_timer = MESSAGE_DISPLAY_TIME
    elif key == ord('3'):
        if 'mustache' in active_filters:
            active_filters.remove('mustache')
        else:
            active_filters.append('mustache')
        status_message = "✅ 필터 토글: 수염"
        message_timer = MESSAGE_DISPLAY_TIME
    elif key == ord('4'):
        if 'crown' in active_filters:
            active_filters.remove('crown')
        else:
            active_filters.append('crown')
        status_message = "✅ 필터 토글: 왕관"
        message_timer = MESSAGE_DISPLAY_TIME
    elif key == ord('0'):
        active_filters = []
        status_message = "✅ 필터 모두 해제"
        message_timer = MESSAGE_DISPLAY_TIME
    elif key in (ord('+'), ord('=')):  # 크기 증가
        SIZE_SCALE = min(SIZE_MAX, round(SIZE_SCALE + SIZE_STEP, 2))
        status_message = f"🔍 크기배율: {SIZE_SCALE:.1f}"
        message_timer = MESSAGE_DISPLAY_TIME
    elif key in (ord('-'), ord('_')):  # 크기 감소
        SIZE_SCALE = max(SIZE_MIN, round(SIZE_SCALE - SIZE_STEP, 2))
        status_message = f"🔍 크기배율: {SIZE_SCALE:.1f}"
        message_timer = MESSAGE_DISPLAY_TIME
    elif key == ord('['):  # 알파 감소
        ALPHA_SCALE = max(ALPHA_MIN, round(ALPHA_SCALE - ALPHA_STEP, 2))
        status_message = f"✨ 알파배율: {ALPHA_SCALE:.1f}"
        message_timer = MESSAGE_DISPLAY_TIME
    elif key == ord(']'):  # 알파 증가
        ALPHA_SCALE = min(ALPHA_MAX, round(ALPHA_SCALE + ALPHA_STEP, 2))
        status_message = f"✨ 알파배율: {ALPHA_SCALE:.1f}"
        message_timer = MESSAGE_DISPLAY_TIME

cap.release()
cv2.destroyAllWindows()