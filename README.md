# FootFrame

FootFrame는 **기존 체스보드가 인쇄된 A4(210 × 297mm)**와 스마트폰 사진으로 발볼·발길이를 추정하는 2D 컴퓨터비전 웹 프로그램입니다. 기존 YOLO 모델로 **발 아치(high / low / normal)를 선택적으로 분류**할 수 있습니다.

## 파일 구성

```text
FootFrame/
├── app.py                     # 유일한 Python 코드: 웹서버 + CV 측정 + YOLO
├── README.md                  # 사용 및 코드 설명을 모두 담은 문서
├── requirements.txt           # OpenCV 기본 실행 라이브러리
├── requirements-yolo.txt      # 선택형 아치 분류 라이브러리
├── render.yaml                # Render 웹 배포 설정
├── .gitignore                 # 로컬 분석 자료·복원된 모델 업로드 방지
└── models/arch/
    ├── best.pt.part01 ~ best.pt.part07  # 기존 YOLO 가중치(각 8MiB 이하)
    └── best.pt.sha256         # 원본 모델 무결성 확인
```

**Python 파일은 `app.py` 하나뿐입니다.** 기존 `core/`의 파이썬 파일은 이 파일 안에 통합했습니다. 개발용 `tests/`, `tools/`, 사진, 검증 이미지는 GitHub 배포본에 포함하지 않습니다.

## 실행 방법

Python **3.11 이상** 환경에서 프로젝트 폴더를 열고 다음을 실행합니다.

```bash
python -m pip install -r requirements.txt
python app.py
```

웹 브라우저에서 `http://localhost:8000`에 접속합니다. 위쪽 사진과 측면 사진을 여러 장 업로드해 분석할 수 있습니다. **새로운 인쇄판, 수동 꼭짓점 선택, 3D 복원은 필요하지 않습니다.** 사진은 서버에서 메모리로 처리하며 별도의 사진 저장 기능은 없습니다.

### 기존 YOLO를 사용할 경우

```bash
python -m pip install -r requirements-yolo.txt
python app.py
```

웹사이트에 **아치 사진 1장**을 추가하면 `app.py`의 YOLO 코드가 기존 모델 조각을 `best.pt`로 복원하고 SHA256을 검사한 뒤 분류를 시도합니다. 발볼·발길이는 별도의 OpenCV 경로로 계산되며 YOLO 분류 결과가 치수 측정값을 변경하지 않습니다. Ultralytics가 없는 서버에서는 아치 분석만 사용할 수 없습니다.

## `app.py`의 코드 구성과 강의 기법

파일 안의 **구역 주석**으로 해당 코드를 찾을 수 있습니다.

| 강의 내용 | 실제 사용 내용 |
|---|---|
| L02 — Image Processing | Grayscale, Gaussian Blur, Thresholding, Adaptive Thresholding, Morphology, Contours, Canny, HoughLinesP |
| L03 — Color | HSV·Lab 기반 종이와 발 영역 분리 |
| L05·L06 — Image Geometry | 체스보드가 인쇄된 A4 후보 검출, `getPerspectiveTransform`, `warpPerspective` |
| L07 — Image Correspondence / Solving Problems | ORB, 특징점 대응, RANSAC 직선 적합, 이상값 제거 및 다중 사진 통합 |
| 기존 YOLO | 아치 high / low / normal 분류만 수행. 기존 모델 가중치 변경 없음 |

대표적인 함수 이름은 `detect_squares()`, `detect_paper()`, `rectify()`, `foot_mask()`, `top_width()`, `side_length()`, `analyze_batch()`, `predict_arch()`, `process_uploads()`입니다. 일반 흰 A4용 예비 코드는 `detect_plain_a4()`처럼 별도 이름을 사용해 체스보드 검출 경로와 충돌하지 않도록 했습니다.

## GitHub 및 Render 배포

1. 이 ZIP의 압축을 해제합니다.
2. **압축 해제된 파일과 폴더**를 GitHub 저장소의 `main` 브랜치 **루트**에 업로드합니다.
3. Render에 저장소를 연결하고 Build Command `pip install -r requirements.txt`, Start Command `python app.py`, Health Check `/health`를 설정합니다. `render.yaml`에 기본 설정이 포함되어 있습니다.
4. YOLO를 실행할 서버라면 Build Command를 `pip install -r requirements-yolo.txt`로 바꾸고 PyTorch/Ultralytics를 실행할 수 있는 CPU·메모리 환경을 준비합니다. Render 무료 인스턴스에서 YOLO 실제 추론은 보장되지 않습니다.

**개인 발 사진, 테스트 코드, 검증 파일은 공개 GitHub에 업로드하지 마세요.** `best.pt`는 첫 YOLO 분석 시 복원되는 파일이므로 `.gitignore`에서 제외되어 있습니다.


### GitHub 웹 업로드 실패 시

GitHub 브라우저는 한 파일당 25MiB까지만 업로드합니다. 기존 모델은 약 52MiB이므로 **`best.pt` 원본을 업로드하지 마세요.** 이 패키지에는 8MiB 이하의 `best.pt.part01`~`part07` 7개가 있고, 실행 시 `app.py`가 SHA256 확인 후 원본 모델을 자동 복원합니다.

- `.zip` 파일 자체를 GitHub에 올리지 말고 압축을 푼 파일들을 업로드합니다.
- **먼저** `app.py`, `README.md`, `requirements*.txt`, `render.yaml`, `.gitignore`를 저장소 루트에 올리고 커밋합니다.
- 다음으로 `models/arch/`의 7개 `best.pt.partNN` 및 `best.pt.sha256`을 폴더 구조를 유지하며 업로드합니다. 업로드 오류가 반복된다면 모델 조각은 2~3개씩 나누어 커밋합니다.
- `best.pt`(복원 모델), 다른 ZIP, 개인 촬영 사진은 올리지 않습니다.

## 검증 상태와 한계

기존 사진 `12290.jpg`~`12298.jpg` 9장으로 통합 이전과 이후의 측정 결과를 비교했으며 **발볼 95.83mm, 발길이 256.0mm**가 동일했습니다. 기존 사용·제외 사진도 동일합니다. 제공하신 실측값 100mm·250mm와의 차이는 각각 **-4.17mm, +6.0mm**입니다. 실측값을 내부 보정 계수로 사용하지 않았습니다.

**이 결과는 잠정적인 A4 평면 투영 치수입니다.** 검은 칸의 바깥 사각형을 실제 A4 외곽으로 확정할 수 없는 사진이 있으며, 발의 높이·촬영 각도에 따른 시차와 다른 사람에 대한 측정 정확도도 검증되지 않았습니다. 신발 사이즈 확정값으로 사용하지 마세요.
