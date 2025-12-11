# AR Face Filter Project 
### 👤 Kim Sejung (201935020)

## 1. 프로젝트 개요 (Project Overview)
This project is a real-time Augmented Reality (AR) application developed using **Python**, **OpenCV**, and **MediaPipe**. The system detects faces and hands through a webcam to apply dynamic digital filters and enable touchless interaction. 

It goes beyond simple overlays by implementing advanced computer vision techniques such as:
*   **Mesh-based Face Distortion** (Liquid warp effects)
*   **Texture Mapping** (Face painting that adheres to skin)
*   **Physics-based Particle Systems** (Interactive elements)
*   **Hand Gesture Recognition** (Touchless UI control)

## 2. 데모 영상 (Demo)
<video src="Sejung/videos/rec_20251211_123859.mp4" width="100%" controls></video>

> *Note: If the video above does not play, please check the file at `Sejung/videos/rec_20251211_123859.mp4`.*

---

## 3. 주요 기능 (Key Features)

### 🎭 Advanced Face Filters
1.  **Face Distortion (Big Eyes)**
    *   Uses localized mesh warping to enlarge eyes in real-time.
    *   Adjustable distortion strength and radius.
2.  **Face Painting (Joker)**
    *   Applies texture directly onto the face mesh (lips, eyes) using polygon rendering.
    *   The makeup moves naturally with facial expressions.
3.  **Interactive Particles**
    *   Opening the mouth triggers a physics-based fire particle system.
4.  **2D Accessories**
    *   Overlays glasses, hats, and crowns aligned with specific facial landmarks.

### 🖐 Gesture Control Interface
Control the application without touching the keyboard:
*   **✌️ V-Sign**: Take a Screenshot
*   **🖐️ Palm**: Navigate Menu (Next Item)
*   **✊ Fist**: Select / Deselect Item
*   **☝️ Index Point**: Change Virtual Background

### 🖼️ Virtual Backgrounds
*   Replaces the real background with **Blur**, **Solid Colors**, or **Pattern Images** using Selfie Segmentation.

---

## 4. 설치 및 패키지 정보 (Installation & Requirements)

### Environment
*   **Python 3.8+**

### Required Packages (with versions)
The following packages are required. You can install them via `pip`.

| Package | Version | Usage |
| :--- | :--- | :--- |
| `opencv-python` | >= 4.5.0 | Image processing & Computer Vision |
| `mediapipe` | >= 0.8.9 | Face Mesh, Hands, Selfie Segmentation |
| `numpy` | >= 1.19.0 | Matrix operations & Geometric calcs |
| `pillow` | >= 8.0.0 | Korean text rendering & Image handling |

**Installation Command:**
```bash
pip install opencv-python mediapipe numpy pillow
```

---

## 5. 실행 방법 (Usage)

This project is located in the `Sejung` directory.

1.  **Navigate to the project directory:**
    ```bash
    cd Sejung
    ```

2.  **Run the main script:**
    ```bash
    python main.py
    ```

### 🎮 Control Guide
| Key | Function |
| :--- | :--- |
| **A / D** | Navigate Menu (Left / Right) |
| **Space** | Toggle Filter |
| **Tab** | Change Background |
| **S** | Save Screenshot |
| **R** | Start/Stop Recording |
| **+ / -** | Adjust Filter Size |
| **[ / ]** | Adjust Filter Transparency |

---

## 6. 참고 자료 (References)
*   **Google MediaPipe Solutions**: [https://developers.google.com/mediapipe/solutions](https://developers.google.com/mediapipe/solutions)
    *   Used for Face Mesh, Hands, and Selfie Segmentation models.
*   **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/)
    *   Referenced for `warpAffine`, `remap`, and image processing functions.
*   **NumPy Documentation**: [https://numpy.org/doc/](https://numpy.org/doc/)
    *   Referenced for vector arithmetic and mask operations.
