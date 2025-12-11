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

# [가상 배경] 세그멘테이션 초기화
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# [제스처 컨트롤] 손 인식 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7, # 제스처 오작동 방지를 위해 정확도 높임
    min_tracking_confidence=0.5
)

# --- [2. 필터 생성 함수들] ---
def create_glasses_filter(width, height):
    """안경 필터 생성"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    lens_radius = min(width, height) // 4
    cv2.circle(img, (width//4, height//2), lens_radius, (50, 50, 50, 200), -1)
    cv2.circle(img, (3*width//4, height//2), lens_radius, (50, 50, 50, 200), -1)
    cv2.circle(img, (width//4, height//2), lens_radius, (0, 0, 0, 255), 3)
    cv2.circle(img, (3*width//4, height//2), lens_radius, (0, 0, 0, 255), 3)
    cv2.line(img, (width//4 - lens_radius, height//2), (0, height//2), (0, 0, 0, 255), 3)
    cv2.line(img, (3*width//4 + lens_radius, height//2), (width, height//2), (0, 0, 0, 255), 3)
    cv2.line(img, (width//4, height//2 - lens_radius//2), (3*width//4, height//2 - lens_radius//2), (0, 0, 0, 255), 3)
    return img

def create_hat_filter(width, height):
    """모자 필터 생성"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.ellipse(img, (width//2, height//3), (width//2, height//3), 0, 0, 180, (139, 69, 19, 255), -1)
    cv2.ellipse(img, (width//2, height//3), (width//2, height//3), 0, 0, 180, (0, 0, 0, 255), 3)
    cv2.rectangle(img, (width//2 - 20, height//3 - 5), (width//2 + 20, height//3 + 5), (255, 0, 0, 255), -1)
    return img

def create_mustache_filter(width, height):
    """수염 필터 생성"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.ellipse(img, (width//2, height//2), (width//3, height//4), 0, 0, 360, (50, 50, 50, 220), -1)
    cv2.ellipse(img, (width//2, height//2), (width//3, height//4), 0, 0, 360, (0, 0, 0, 255), 2)
    cv2.ellipse(img, (width//4, height//2), (width//8, height//6), 0, 0, 360, (50, 50, 50, 220), -1)
    cv2.ellipse(img, (3*width//4, height//2), (width//8, height//6), 0, 0, 360, (50, 50, 50, 220), -1)
    return img

def create_crown_filter(width, height):
    """왕관 필터 생성"""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    points = np.array([
        [width//2, 0], [width//2 - width//4, height//2], [width//4, height//2],
        [width//2, height//3], [3*width//4, height//2], [width//2 + width//4, height//2]
    ], np.int32)
    cv2.fillPoly(img, [points], (255, 215, 0, 255))
    cv2.polylines(img, [points], True, (0, 0, 0, 255), 2)
    cv2.circle(img, (width//2, height//6), 5, (255, 0, 0, 255), -1)
    cv2.circle(img, (width//4, height//3), 4, (0, 255, 0, 255), -1)
    cv2.circle(img, (3*width//4, height//3), 4, (0, 0, 255, 255), -1)
    return img

def distort_region(image, cx, cy, radius, strength=0.5):
    """영역 왜곡 (왕눈이 효과)"""
    try:
        h, w = image.shape[:2]
        x1, x2 = max(0, cx - radius), min(w, cx + radius)
        y1, y2 = max(0, cy - radius), min(h, cy + radius)
        if x2 <= x1 or y2 <= y1: return image
        
        roi = image[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        grid_y, grid_x = np.indices((rh, rw), dtype=np.float32)
        
        rel_cx, rel_cy = cx - x1, cy - y1
        r = np.sqrt((grid_x - rel_cx)**2 + (grid_y - rel_cy)**2)
        mask = r < radius
        
        map_x, map_y = grid_x.copy(), grid_y.copy()
        factor = 1.0 - strength * (1.0 - r[mask] / radius)
        map_x[mask] = (grid_x[mask] - rel_cx) * factor + rel_cx
        map_y[mask] = (grid_y[mask] - rel_cy) * factor + rel_cy
        
        distorted = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        np.copyto(roi, distorted, where=np.stack((mask,)*3, axis=-1))
        image[y1:y2, x1:x2] = roi
    except: pass
    return image

# --- [3. 핵심 함수: 투명 이미지 합성] ---
def overlay_transparent(background, overlay, x, y):
    try:
        bg_h, bg_w, _ = background.shape
        h, w, _ = overlay.shape

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

        alpha = overlay[:, :, 3] / 255.0
        colors = overlay[:, :, :3]
        roi = background[y:y+h, x:x+w]
        
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1.0 - alpha) + colors[:, :, c] * alpha
            
        background[y:y+h, x:x+w] = roi
        return background
    except Exception as e:
        return background

# --- [4. 이미지 파일 로드] ---
def load_filter_image(filename, default_width=300, default_height=100, create_func=None):
    current_dir = os.getcwd()
    possible_paths = [filename, os.path.join('Sejung', filename), os.path.join('..', filename)]
    img = None
    for path in possible_paths:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"✅ 이미지 로드 성공: {path}")
                break
    if img is None and create_func:
        print(f"⚠️ '{filename}' 대체 코드 사용")
        img = create_func(default_width, default_height)
    elif img is not None and img.shape[2] < 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

print("\n=== 리소스 로드 중 ===")
glasses_img = load_filter_image('glasses.png', 300, 100, create_glasses_filter)
hat_img = load_filter_image('hat.png', 300, 180, create_hat_filter)
mustache_img = load_filter_image('mustache.png', 300, 150, create_mustache_filter)
crown_img = load_filter_image('crown.png', 300, 240, create_crown_filter)
print("=" * 30 + "\n")

# --- [5. UI 및 텍스트 관련] ---
def put_korean_text(img, text, position, font_size=30, color=(0, 255, 0), align='left'):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        font = None
        font_paths = ["C:/Windows/Fonts/malgun.ttf", "malgun.ttf", "gulim.ttc"]
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            except: continue
        if font is None: font = ImageFont.load_default()

        # 정렬 처리
        x, y = position
        if align == 'center':
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = x - text_width // 2
        
        draw.text((x, y), text, font=font, fill=color)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except: pass
    return img

# --- [6. 필터 설정] ---
FILTER_SETTINGS = {
    'glasses': {'size_ratio': 2.3, 'height_ratio': 0.4, 'offset_x': 0, 'offset_y': 0},
    'hat': {'size_ratio': 2.5, 'height_ratio': 0.6, 'offset_x': 0, 'offset_y': 60},
    'mustache': {'size_ratio': 1.8, 'height_ratio': 0.5, 'offset_x': 0, 'offset_y': -25},
    'crown': {'size_ratio': 2.2, 'height_ratio': 0.8, 'offset_x': 0, 'offset_y': 15}
}

SIZE_SCALE = 1.0
ALPHA_SCALE = 1.0
SIZE_STEP = 0.1
ALPHA_STEP = 0.1
SIZE_MIN, SIZE_MAX = 0.5, 3.0
ALPHA_MIN, ALPHA_MAX = 0.1, 2.0

SCREENSHOT_DIR = "screenshots"
SCREENSHOT_FMT = "jpg"
SCREENSHOT_QUALITY = 95
RECORD_DIR = "videos"
RECORD_CODEC = "mp4v"
RECORD_FPS_FALLBACK = 30

# --- [7. 상태 관리] ---
FILTER_ITEMS = ['glasses', 'hat', 'mustache', 'crown', 'big_eyes']
FILTER_LABELS = ['안경', '모자', '수염', '왕관', '왕눈이']
current_cursor_index = 0
active_filters = ['glasses']
background_mode = 0
BACKGROUND_MODES = {0: "없음", 1: "블러", 2: "핑크", 3: "밤하늘"}

# 제스처 쿨타임 관리
gesture_cooldown = 0
GESTURE_LOCK_TIME = 20 # 프레임

# --- [제스처 인식 함수] ---
def detect_gesture(hand_landmarks):
    """
    손 랜드마크를 분석하여 제스처 반환
    Return: 'V', 'PALM', 'FIST', 'POINT', 'NONE'
    """
    # 손가락 끝(TIP)과 마디(PIP) 인덱스
    # 검지(8), 중지(12), 약지(16), 소지(20)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    # 펴짐 여부 확인 (화면 좌표계: 위가 y 작음)
    is_open = []
    for tip, pip in zip(tips, pips):
        is_open.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    
    # 제스처 판별
    # 1. 보(PALM): 4개 모두 펴짐 -> 다음 메뉴 이동
    if all(is_open):
        return "PALM"
    # 2. 주먹(FIST): 4개 모두 접힘 -> 선택/해제
    if not any(is_open):
        return "FIST"
    # 3. 브이(V): 검지, 중지 펴짐 + 나머지 접힘 -> 스크린샷
    if is_open[0] and is_open[1] and not is_open[2] and not is_open[3]:
        return "V"
    # 4. 검지(POINT): 검지 펴짐 + 나머지 접힘 -> 배경 변경
    if is_open[0] and not is_open[1] and not is_open[2] and not is_open[3]:
        return "POINT"
        
    return "NONE"

# --- [배경 적용] ---
def apply_background(image, mode):
    if mode == 0: return image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(image_rgb)
    mask = results.segmentation_mask
    condition = np.stack((mask,) * 3, axis=-1) > 0.5
    
    h, w, c = image.shape
    if mode == 1:
        bg_image = cv2.GaussianBlur(image, (55, 55), 0)
    elif mode == 2:
        bg_image = np.zeros((h, w, 3), dtype=np.uint8)
        bg_image[:] = (180, 105, 255)
    elif mode == 3:
        bg_image = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            color_val = int(100 * (i / h))
            bg_image[i, :] = (50, 20 + color_val, 0)
        np.random.seed(42)
        for _ in range(50):
            cv2.circle(bg_image, (np.random.randint(0, w), np.random.randint(0, h)), 
                      np.random.randint(1, 3), (255, 255, 255), -1)
    
    return np.where(condition, image, bg_image) if 'bg_image' in locals() else image

# --- [필터 적용] ---
def apply_filter(image, face_landmarks, filter_type, h, w):
    if filter_type == 'none': return image
    
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)
    
    dx, dy = rx - lx, ry - ly
    angle = np.degrees(np.arctan2(dy, dx))
    eye_dist = np.sqrt(dx**2 + dy**2)
    
    settings = FILTER_SETTINGS.get(filter_type, {})
    size_ratio = settings.get('size_ratio', 2.0) * SIZE_SCALE
    height_ratio = settings.get('height_ratio', 0.5)
    offset_x, offset_y = settings.get('offset_x', 0), settings.get('offset_y', 0)
    
    filter_img = None
    target_width = int(eye_dist * size_ratio)
    
    if filter_type == 'glasses':
        if glasses_img is not None:
            scale = target_width / glasses_img.shape[1]
            filter_img = cv2.resize(glasses_img.copy(), (target_width, int(glasses_img.shape[0] * scale)))
        else: filter_img = create_glasses_filter(target_width, int(target_width * height_ratio))
        center_x = (lx + rx) // 2 - target_width // 2 + offset_x
        center_y = (ly + ry) // 2 - filter_img.shape[0] // 2 + offset_y
        
    elif filter_type == 'hat':
        forehead = face_landmarks.landmark[10]
        fx, fy = int(forehead.x * w), int(forehead.y * h)
        if hat_img is not None:
            scale = target_width / hat_img.shape[1]
            filter_img = cv2.resize(hat_img.copy(), (target_width, int(hat_img.shape[0] * scale)))
        else: filter_img = create_hat_filter(target_width, int(target_width * height_ratio))
        center_x = fx - target_width // 2 + offset_x
        center_y = fy - filter_img.shape[0] + offset_y

    elif filter_type == 'mustache':
        nose = face_landmarks.landmark[4]
        lip = face_landmarks.landmark[13]
        nx, ny = int(nose.x * w), int(nose.y * h)
        ux, uy = int(lip.x * w), int(lip.y * h)
        if mustache_img is not None:
            scale = target_width / mustache_img.shape[1]
            filter_img = cv2.resize(mustache_img.copy(), (target_width, int(mustache_img.shape[0] * scale)))
        else: filter_img = create_mustache_filter(target_width, int(target_width * height_ratio))
        center_x = (nx + ux) // 2 - target_width // 2 + offset_x
        center_y = (ny + uy) // 2 + offset_y

    elif filter_type == 'crown':
        forehead = face_landmarks.landmark[10]
        fx, fy = int(forehead.x * w), int(forehead.y * h)
        if crown_img is not None:
            scale = target_width / crown_img.shape[1]
            filter_img = cv2.resize(crown_img.copy(), (target_width, int(crown_img.shape[0] * scale)))
        else: filter_img = create_crown_filter(target_width, int(target_width * height_ratio))
        center_x = fx - target_width // 2 + offset_x
        center_y = fy - filter_img.shape[0] + offset_y

    elif filter_type == 'big_eyes':
        le = face_landmarks.landmark[468] # 왼쪽 홍채
        re = face_landmarks.landmark[473] # 오른쪽 홍채
        lx, ly = int(le.x * w), int(le.y * h)
        rx, ry = int(re.x * w), int(re.y * h)
        
        eye_dist = np.sqrt((lx-rx)**2 + (ly-ry)**2)
        radius = int(eye_dist * 0.45 * SIZE_SCALE)
        strength = 0.6 * ALPHA_SCALE
        
        image = distort_region(image, lx, ly, radius, strength)
        image = distort_region(image, rx, ry, radius, strength)
        return image

    if filter_img is not None:
        M = cv2.getRotationMatrix2D((filter_img.shape[1]//2, filter_img.shape[0]//2), -angle, 1)
        rotated = cv2.warpAffine(filter_img, M, (filter_img.shape[1], filter_img.shape[0]))
        if rotated.shape[2] == 4 and ALPHA_SCALE != 1.0:
            rotated[:, :, 3] = np.clip(rotated[:, :, 3] * ALPHA_SCALE, 0, 255)
        image = overlay_transparent(image, rotated, center_x, center_y)
        
    return image

def apply_filters(image, face_landmarks, filters, h, w):
    for f in filters: image = apply_filter(image, face_landmarks, f, h, w)
    return image

def save_screenshot(image, filter_name):
    try:
        if not os.path.exists(SCREENSHOT_DIR): os.makedirs(SCREENSHOT_DIR)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SCREENSHOT_DIR, f"screenshot_{filter_name}_{timestamp}.{SCREENSHOT_FMT}")
        params = [cv2.IMWRITE_JPEG_QUALITY, SCREENSHOT_QUALITY] if SCREENSHOT_FMT in ('jpg','jpeg') else []
        if cv2.imwrite(filename, image, params): return filename
    except: pass
    return None

# --- [UX 개선: 하단 인벤토리 UI 그리기] ---
def draw_ui(image, h, w):
    # 하단 반투명 바 (높이 증가: 80 -> 100)
    overlay = image.copy()
    bar_height = 100
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    
    # 아이템 간격 계산
    item_count = len(FILTER_ITEMS)
    spacing = w // (item_count + 1)
    
    for i, (item_key, label) in enumerate(zip(FILTER_ITEMS, FILTER_LABELS)):
        x_pos = spacing * (i + 1)
        # 아이템 위치 조금 위로 조정
        y_pos = h - bar_height // 2 - 5
        
        # 1. 커서 표시 (선택된 아이템 강조)
        if i == current_cursor_index:
            # 커서 박스
            box_w, box_h = 100, 60
            cv2.rectangle(image, (x_pos - box_w//2, y_pos - box_h//2), 
                         (x_pos + box_w//2, y_pos + box_h//2), (255, 255, 0), 2)
            
        # 2. 활성 상태 표시 (착용 중인 아이템)
        is_active = item_key in active_filters
        text_color = (0, 255, 0) if is_active else (150, 150, 150) # 착용:초록, 미착용:회색
        
        # 텍스트 그리기 (중앙 정렬)
        image = put_korean_text(image, label, (x_pos, y_pos - 15), 
                               font_size=24, color=text_color, align='center')
        
        # ON/OFF 상태 텍스트
        status_text = "ON" if is_active else "OFF"
        status_color = (0, 255, 0) if is_active else (100, 100, 100)
        image = put_korean_text(image, status_text, (x_pos, y_pos + 20),
                               font_size=16, color=status_color, align='center')

    # 상단 정보 표시
    top_info = f"배경: {BACKGROUND_MODES[background_mode]}"
    image = put_korean_text(image, top_info, (20, 20), font_size=20, color=(255, 255, 255))
    
    # 제스처 가이드 추가
    gesture_guide = "손동작: ✌(촬영) 🖐(이동) ✊(선택) ☝(배경)"
    image = put_korean_text(image, gesture_guide, (w - 20, 70), font_size=18, color=(255, 200, 100), align='right')

    # 조작 가이드 (위치 상향 조정)
    guide = "이동:[A/D]  선택:[SPACE]  배경:[TAB]  크기:[+/-]  투명도:[ [/] ]  촬영:[S]  녹화:[R]"
    image = put_korean_text(image, guide, (w//2, h - 25), font_size=16, color=(200, 200, 200), align='center')
    
    return image

# --- [8. 메인 실행] ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('AR Filter Project', cv2.WINDOW_NORMAL)
cv2.resizeWindow('AR Filter Project', 1280, 720)

status_message = ""
message_timer = 0
recording = False
video_writer = None

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # 성능 최적화: 이미지 쓰기 금지
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Face Mesh & Hands 처리
    results_face = face_mesh.process(image)
    results_hands = hands.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if background_mode != 0: image = apply_background(image, background_mode)
    
    h, w, c = image.shape
    
    # 얼굴 필터 처리
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            image = apply_filters(image, face_landmarks, active_filters, h, w)
            # 입 벌림 효과
            top_y = face_landmarks.landmark[13].y
            bot_y = face_landmarks.landmark[14].y
            if int(abs(top_y - bot_y) * h) > 40:
                cv2.putText(image, "Wow!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

    # 제스처 처리
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # 손 랜드마크 그리기 (디버깅용)
            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 쿨타임 체크
            if gesture_cooldown == 0:
                gesture = detect_gesture(hand_landmarks)
                
                if gesture != "NONE":
                    if gesture == "V": # 스크린샷
                        f_name = "_".join(active_filters) if active_filters else "none"
                        if save_screenshot(image, f_name):
                            status_message = "✌ 찰칵!"
                        gesture_cooldown = GESTURE_LOCK_TIME * 2 # 촬영은 쿨타임 길게
                        
                    elif gesture == "PALM": # 이동 (오른쪽으로)
                        current_cursor_index = (current_cursor_index + 1) % len(FILTER_ITEMS)
                        status_message = "🖐 이동"
                        gesture_cooldown = GESTURE_LOCK_TIME
                        
                    elif gesture == "FIST": # 선택/해제
                        selected_item = FILTER_ITEMS[current_cursor_index]
                        if selected_item in active_filters:
                            active_filters.remove(selected_item)
                            status_message = "✊ 해제"
                        else:
                            active_filters.append(selected_item)
                            status_message = "✊ 장착"
                        gesture_cooldown = GESTURE_LOCK_TIME
                        
                    elif gesture == "POINT": # 배경 변경
                        background_mode = (background_mode + 1) % len(BACKGROUND_MODES)
                        status_message = f"☝ 배경: {BACKGROUND_MODES[background_mode]}"
                        gesture_cooldown = GESTURE_LOCK_TIME
                    
                    message_timer = 30

    # 쿨타임 감소
    if gesture_cooldown > 0:
        gesture_cooldown -= 1

    # UI 그리기
    image = draw_ui(image, h, w)
    
    # 상태 메시지
    if status_message and message_timer > 0:
        image = put_korean_text(image, status_message, (w//2, h//2), font_size=40, color=(0, 255, 255), align='center')
        message_timer -= 1
        
    if recording:
        # 녹화 표시 위치 우측 상단으로 변경
        cv2.circle(image, (w - 100, 40), 10, (0, 0, 255), -1)
        image = put_korean_text(image, "REC", (w - 70, 30), font_size=20, color=(0, 0, 255))
        if video_writer: video_writer.write(image)

    cv2.imshow('AR Filter Project', image)
    
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): break
    
    # --- [UX 조작 키 매핑] ---
    elif key == ord('a') or key == ord('A'):
        current_cursor_index = (current_cursor_index - 1) % len(FILTER_ITEMS)
    elif key == ord('d') or key == ord('D'):
        current_cursor_index = (current_cursor_index + 1) % len(FILTER_ITEMS)
    elif key == ord(' '):
        selected_item = FILTER_ITEMS[current_cursor_index]
        if selected_item in active_filters:
            active_filters.remove(selected_item)
            status_message = "OFF"
        else:
            active_filters.append(selected_item)
            status_message = "ON"
        message_timer = 20
    
    elif key == 9: # Tab
        background_mode = (background_mode + 1) % len(BACKGROUND_MODES)
        status_message = f"배경: {BACKGROUND_MODES[background_mode]}"
        message_timer = 30
        
    # 기존 기능 키들
    elif key == ord('s') or key == ord('S'):
        f_name = "_".join(active_filters) if active_filters else "none"
        if save_screenshot(image, f_name):
            status_message = "저장 완료!"
            message_timer = 30
    elif key == ord('r') or key == ord('R'):
        if not recording:
            if not os.path.exists(RECORD_DIR): os.makedirs(RECORD_DIR)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(RECORD_DIR, f"rec_{ts}.mp4")
            video_writer = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*RECORD_CODEC), 30, (w, h))
            recording = True
            status_message = "녹화 시작"
        else:
            recording = False
            if video_writer: video_writer.release()
            video_writer = None
            status_message = "녹화 저장됨"
        message_timer = 30
    elif key == ord('0'):
        active_filters = []
        background_mode = 0
        status_message = "초기화"
        message_timer = 30
    elif key in (ord('+'), ord('=')): SIZE_SCALE = min(SIZE_MAX, SIZE_SCALE + SIZE_STEP)
    elif key in (ord('-'), ord('_')): SIZE_SCALE = max(SIZE_MIN, SIZE_SCALE - SIZE_STEP)
    elif key == ord(']'): ALPHA_SCALE = min(ALPHA_MAX, ALPHA_SCALE + ALPHA_STEP)
    elif key == ord('['): ALPHA_SCALE = max(ALPHA_MIN, ALPHA_SCALE - ALPHA_STEP)

cap.release()
if video_writer: video_writer.release()
cv2.destroyAllWindows()
