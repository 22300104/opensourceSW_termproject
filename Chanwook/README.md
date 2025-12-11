# Lee Chanwook (202033757)




## Tree Segmentation (벚꽃 나무 세그멘테이션)

간단한 이미지 세그멘테이션을 통해 나무/꽃 영역만 컬러로 남기고 배경을 흑백으로 만드는 프로젝트입니다.

## 1. 프로젝트 개요

이 프로젝트는 캠퍼스에서 찍은 **벚꽃 나무 사진**에서  
나무 + 꽃 영역만 컬러로 유지하고, 나머지 배경은 흑백으로 만드는  
간단한 **이미지 세그멘테이션(Tree Segmentation)** 프로젝트입니다.

- 입력: `examples/input/campus_trees.jpg` (벚꽃 나무 사진)
- 출력:
  - 나무 + 꽃 영역 마스크 (흑백 이미지)
  - 나무 + 꽃만 컬러로 남기고 배경을 흑백으로 만든 결과 이미지

OpenCV의 색 공간 변환(HSV), 엣지 검출, 이진화(Threshold)를 이용해  
하늘 영역을 제거하고, 나무/꽃 영역만 남기도록 마스크를 생성한 뒤  
원본 이미지와 합성해서 결과를 만듭니다.

---

## 2. 데모 / 예시 이미지

### (1) 입력 이미지

벚꽃 나무가 포함된 원본 이미지입니다.

![Original](images/original_campus_trees.png)

---

### (2) 나무 + 꽃 마스크 (흑백)

나무와 꽃이 있는 영역을 흰색(255), 나머지는 검은색(0)으로 나타낸 마스크입니다.

![Mask (Tree + Blossoms)](images/mask_campus_trees.png)

---

### (3) 최종 Tree Segmentation 결과

마스크를 이용해 **나무 + 꽃 영역은 컬러**, 나머지 배경은 **흑백**으로 만든 결과입니다.

![Tree Segmentation](images/segmentation_campus_trees.png)

---

## 3. 사용한 패키지와 버전

- Python 3.10
- OpenCV (opencv-python)
- NumPy
- Matplotlib

```
pip install -r requirements.txt
```
위 명령 한 줄로 필요한 패키지를 한 번에 설치할 수 있습니다.


## 4. 실행 방법

프로젝트 루트에서 아래 명령을 실행하세요:

```
pip install -r requirements.txt
python main.py
```

## 5. 참고 자료

OpenCV 공식 문서 – Image Processing, Color Spaces, Thresholding

수업 시간에 제공된 OpenCV 예제 코드 및 강의 자료

위 자료들을 참고해 색 공간 변환(HSV), 엣지 검출(Laplacian),
이진화(threshold)와 마스크 합성 방법을 응용하여 구현했습니다.
