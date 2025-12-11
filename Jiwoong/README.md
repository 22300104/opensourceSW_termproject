# Yun JiWoong (201935088)

---

# Pyramid vs Direct Blending (OpenCV)

이 프로젝트는 Gaussian Pyramid와 Laplacian Pyramid를 이용한 이미지 블렌딩(Pyramid blending)과  
단순 절반 병합 방식(Direct blending)을 비교하여 시각적으로 어떤 차이가 있는지 보여주는 간단한 오픈소스 SW입니다.

---

## Features
- OpenCV 기반 Pyramid blending 구현
- Direct blending과 결과 비교
- 이미지 크기 자동 조정 기능
- 명령행 실행 방식 제공

---

## How to Run

### 1. Install dependencies

```pip install -r opencv-python```
```pip install -r numpy```

### 2. Run blending

```python pyramid_blending.py --img1 examples/apple.jpg --img2 examples/orange.jpg```


결과는 `outputs/` 폴더에 저장됩니다.
- Pyramid_blending.jpg  
- Direct_blending.jpg  

---

## 🖼 예시 결과

<img width="2080" height="1098" alt="1" src="https://github.com/user-attachments/assets/33037e9c-17e6-4c70-87f3-2cd2e97cec76" />

---

## Reference
- OpenCV Python Tutorials
- Laplacian Pyramid Blending (Burt and Adelson, 1983)
