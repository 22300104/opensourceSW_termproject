# AR Face Filter Project

## 👥 Team Members
*   **Kim Sejung (201935020)** - AR Face Filter & Gesture Control
*   *(Other Team Member Name)* - *(Role/Project)*
*   *(Other Team Member Name)* - *(Role/Project)*

---

## 👤 Kim Sejung (201935020)

### 1. Project Overview
This module is a real-time Augmented Reality (AR) application developed using Python, OpenCV, and MediaPipe. The system detects faces and hands through a webcam to apply dynamic digital filters and enable touchless interaction. It demonstrates advanced computer vision techniques including face mesh mapping, selfie segmentation, and hand gesture recognition.

### 2. Demo Video
<video src="Sejung/videos/rec_20251211_123859.mp4" width="640" controls></video>

*(If the video does not play, please check the `Sejung/videos/rec_20251211_123859.mp4` file directly.)*

### 3. Key Features
**Advanced Face Filters**
*   **Face Distortion**: Real-time mesh manipulation (e.g., Big Eyes effect) using localized warping algorithms.
*   **Face Painting**: Texture mapping that adheres to facial contours (e.g., Joker makeup) using polygon rendering on face landmarks.
*   **Particle Systems**: Dynamic physics-based particles triggered by facial expressions (e.g., fire effects when opening the mouth).
*   **2D Accessories**: Overlay of graphical elements (glasses, hats, crowns) aligned with specific facial landmarks.

**Gesture Control Interface**
*   **V-Sign**: Capture screenshot.
*   **Open Palm**: Navigate through filter options.
*   **Fist**: Toggle the selected filter on/off.
*   **Index Finger**: Cycle through virtual backgrounds.

**Virtual Backgrounds**
*   Real-time subject segmentation to replace the background with blur effects, solid colors, or dynamic patterns.

### 4. Installation & Usage (Sejung)

1.  **Move to directory**:
    ```bash
    cd Sejung
    ```

2.  **Run the application**:
    ```bash
    python main.py
    ```

### 5. Control Guide
| Input Method | Action | Function |
| :--- | :--- | :--- |
| **Keyboard** | `A` / `D` | Navigate filter menu (Left/Right) |
| | `Space` | Toggle selected filter |
| | `Tab` | Change virtual background |
| | `S` | Save screenshot |
| | `R` | Start/Stop video recording |
| **Gestures** | **V-Sign** | Take screenshot |
| | **Palm** | Navigate menu |
| | **Fist** | Select/Deselect item |
| | **Index Point** | Change background |

---

## 👤 (Next Team Member Name)
*(Placeholder for next member's project description)*

---

## Requirements
The project requires **Python 3.8** or higher.
The following packages are necessary for execution:

*   `opencv-python` (>= 4.5.0)
*   `mediapipe` (>= 0.8.9)
*   `numpy` (>= 1.19.0)
*   `pillow` (>= 8.0.0)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/[your-repository-url].git
    ```

2.  Install the required dependencies:
    ```bash
    pip install opencv-python mediapipe numpy pillow
    ```

## References
*   [Google MediaPipe Solutions](https://developers.google.com/mediapipe/solutions) - Used for Face Mesh, Hands, and Selfie Segmentation.
*   [OpenCV Documentation](https://docs.opencv.org/) - Used for image processing and computer vision tasks.
*   [NumPy Documentation](https://numpy.org/doc/) - Used for matrix operations and geometric calculations.

---
*This project was developed for educational purposes to demonstrate the capabilities of computer vision libraries.*
