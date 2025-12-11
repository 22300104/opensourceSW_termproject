# 👤 Lee JoonKyo (201935111)
## ColorPaletteGen 🎨  
이미지 기반 컬러 팔레트 생성기

## 1. 프로젝트 개요 (Overview)

**ColorPaletteGen**은 이미지에서 대표 색상들을 추출하여  
- RGB / HEX 코드 목록  
- 컬러 팔레트 이미지  

를 자동으로 생성해주는 오픈소스 도구입니다.  

디자이너, 개발자, 프레젠테이션 제작자 등  
이미지 기반 색 조합이 필요한 사용자를 위해 설계된 간단한 CLI 도구입니다.

---

## 2. 데모 (Demo)

아래는 예시 형식입니다.  

### 입력 이미지 예시
`images/input/sample1.jpg`
![sample1](images/input/sample1.jpg)

### 출력 팔레트 예시
`images/result/palette_sample1.png`
![palette_sample1](images/result/sample1_palette.png)

### 터미널 출력 예시

```
$ python -m src.cli --image images/input/sample1.jpg --k 5 --output images/result/sample1_palette.png

=== Extracted Colors ===
1: RGB=(26, 115, 165), HEX=#1A73A5
2: RGB=(215, 234, 248), HEX=#D7EAF8
3: RGB=(53, 148, 190), HEX=#3594BE
4: RGB=(158, 202, 229), HEX=#9ECAE5
5: RGB=(96, 180, 214), HEX=#60B4D6
Palette saved to: images/result/sample1_palette.png
```

---

## 3. 설치 방법 (Installation)

### 요구 환경
- Python 3.10 이상 권장
- pip

### 의존성 설치

```
pip install -r requirements.txt
```

사용 라이브러리:
- opencv-python  
- numpy  
- scikit-learn  
- Pillow  

---

## 4. 실행 방법 (Usage)

### 기본 실행

```
python -m src.cli --image examples/sample1.jpg
```

- 기본 색상 개수: 5  
- 출력 파일: `palette.png`

### 옵션 포함 실행

```
python -m src.cli --image examples/sample1.jpg --k 8 --output out.png --json colors.json
```

#### 옵션 설명
| 옵션 | 설명 |
|------|------|
| `--image` | 입력 이미지 경로 (필수) |
| `--k` | 추출할 대표 색상 개수 (기본 5) |
| `--output` | 팔레트 이미지 저장 경로 |
| `--json` | 색상 정보 JSON 저장 경로 |

---

## 5. 프로젝트 구조 (Project Structure)

```
color-palette-gen/
├─ src/
│  ├─ __init__.py
│  ├─ palette_extractor.py
│  ├─ palette_image.py
│  └─ cli.py
images/
├── input/
│   ├── sample1.jpg
│   ├── sample2.png
│   └── ...
├── result/
│   ├── palette_sample1.png
│   ├── palette_sample2.png
│   └── ...
├─ requirements.txt
└─ README.md
```

---

## 6. 내부 동작 방식 (How It Works)

1. OpenCV로 이미지를 읽고 RGB로 변환  
2. 이미지를 픽셀 배열로 변환  
3. K-Means로 대표 색상 중심(centroid) 추출  
4. RGB → HEX 변환  
5. Pillow로 팔레트 이미지 생성  

---

## 7. 사용 예시 (Examples)

### 5개 색상 추출

```
python -m src.cli --image images/input/sample1.jpg --k 5 --output images/result/sample1_palette.png
```

### 8개 색상 + JSON 저장

```
python -m src.cli --image images/input/sample1.jpg --k 8 --json colors.json --output images/result/sample1_palette.png
```
