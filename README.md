# opencv
OpenCV에 대한 설명을 GitHub의 `md` 파일 형식으로 정리해보겠습니다! Markdown 형식은 간단하고 깔끔하게 문서화할 수 있어서 코드 예시나 설명을 추가할 때 유용해요. 아래는 `README.md` 파일 형태로 OpenCV를 소개하는 예시입니다.

---

````markdown
# OpenCV (Open Source Computer Vision Library)

OpenCV는 이미지 처리와 컴퓨터 비전 작업을 위한 오픈 소스 라이브러리입니다. 이 라이브러리는 다양한 이미지 처리 기술을 제공하며, 영상 처리 및 분석, 객체 인식, 얼굴 인식 등의 작업에 많이 사용됩니다.

## 설치 방법

OpenCV를 설치하려면, Python의 `pip` 패키지 관리자를 사용합니다. 아래 명령어로 OpenCV를 설치할 수 있습니다.

```bash
pip install opencv-python
````

기본적인 OpenCV 패키지 외에도, 추가적인 기능이 포함된 `opencv-contrib-python`을 설치할 수 있습니다.

```bash
pip install opencv-contrib-python
```

## OpenCV 기본 사용법

OpenCV를 사용하여 간단한 이미지 처리 작업을 할 수 있습니다. 아래는 몇 가지 기본적인 사용 예시입니다.

### 1. 이미지 읽기 및 표시

OpenCV에서 이미지를 읽고 표시하는 방법입니다.

```python
import cv2

# 이미지 파일 읽기
image = cv2.imread('image.jpg')

# 이미지 창으로 표시
cv2.imshow('Image', image)

# 키를 누르면 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. 이미지 저장

이미지를 다른 형식으로 저장하는 방법입니다.

```python
import cv2

# 이미지 파일 읽기
image = cv2.imread('image.jpg')

# 이미지 저장
cv2.imwrite('saved_image.jpg', image)
```

### 3. 영상 캡처

웹캠으로 실시간 영상을 캡처하는 방법입니다.

```python
import cv2

# 카메라 열기 (기본 카메라는 0번)
cap = cv2.VideoCapture(0)

while True:
    # 실시간으로 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        break

    # 화면에 프레임 표시
    cv2.imshow('Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라와 창 닫기
cap.release()
cv2.destroyAllWindows()
```

### 4. 이미지 회전

이미지를 회전하는 방법입니다.

```python
import cv2

# 이미지 읽기
image = cv2.imread('image.jpg')

# 이미지 크기 얻기
height, width = image.shape[:2]

# 회전 행렬 생성 (45도 회전)
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)

# 이미지 회전
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# 회전된 이미지 표시
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 주요 기능

OpenCV는 매우 다양한 기능을 제공합니다. 그 중 일부는 다음과 같습니다.

### 1. 얼굴 인식

OpenCV는 얼굴 인식 기능을 내장하고 있습니다. Haar Cascade 분류기를 사용하여 얼굴을 감지할 수 있습니다.

```python
import cv2

# 얼굴 인식 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 읽기
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 인식
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 얼굴에 사각형 표시
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 결과 이미지 표시
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. 객체 추적

OpenCV는 다양한 객체 추적 알고리즘을 제공합니다. 예를 들어, `KCF`, `MIL`, `TLD`, `BOOSTING` 등 여러 알고리즘을 사용할 수 있습니다.

```python
import cv2

# 비디오 파일 또는 웹캠에서 객체 추적
cap = cv2.VideoCapture(0)

# 첫 번째 프레임 읽기
ret, frame = cap.read()

# 추적할 객체 선택
bbox = cv2.selectROI(frame, fromCenter=False, showCrosshair=True)

# 추적 객체 생성
tracker = cv2.TrackerKCF_create()
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()

    # 객체 추적
    ret, bbox = tracker.update(frame)

    if ret:
        # 객체 위치 표시
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # 화면에 프레임 표시
    cv2.imshow('Object Tracking', frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

