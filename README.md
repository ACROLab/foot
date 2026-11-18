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

웹 브라우저에서 `http://localhost:8000`에 접속합니다. 위쪽 사진과 측면 사진을 여러 장 업로드해 분석할 수 있습니다. YOLO 아치 사진은 단독 업로드도 가능합니다. **새로운 인쇄판, 수동 꼭짓점 선택, 3D 복원은 필요하지 않습니다.** 사진은 서버에서 메모리로 처리하며 별도의 사진 저장 기능은 없습니다.

### YOLO 아치 분류 (기존 모델 그대로)

- `best.pt`는 **YOLOv8m-seg 아치 세그멘테이션 모델**입니다. 원본 가중치의 클래스 이름은 Low / High / Normal입니다. 파이썬에서 모델의 추론 경로는 별도이며, 발 길이·발볼 측정값에는 영향을 주지 않습니다.
- 이제 **발볼·발길이 사진 없이 아치 사진 하나만 올려** `발 아치 사진만 YOLO로 분석` 버튼으로 시험할 수 있습니다.
- `/arch-status`: 파일 조각·설치된 라이브러리·메모리 용량 검사. `/analyze-arch`: 실제 YOLO 아치 사진 한 장 업로드 (추론이 불가능한 경우에는 오류 원인을 표시).
- Ultralytics NumPy 사진 입력은 **BGR**을 사용합니다. 이전에는 RGB로 한 번 더 바꿔 잘못 전달했던 부분을 수정했습니다.
- YOLO 가중치의 분할 파일 SHA256 검사는 유지합니다.

#### Render에서 YOLO를 실제로 사용하려면

**Render Free의 메모리는 512MB이므로 이 55MB YOLOv8m-seg+PyTorch 모델을 안정적으로 실행할 수 있다고 보장하지 않습니다.** 기본 `render.yaml`은 무료 2D 측정 사이트를 보호하기 위해 OpenCV만 설치합니다. 따라서 기본 무료 배포에서는 YOLO 상태가 미설치로 표시되는 것이 정상입니다.

YOLO 추론을 실행할 서버(최소 2GB RAM을 권장하며 모델 로딩 피크에 따라 더 필요할 수 있음)에서는 다음을 설정합니다.

1. Render 인스턴스를 충분한 메모리의 플랜 또는 별도 서버로 설정합니다.
2. **Build Command**를 `pip install -r requirements.txt && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision && pip install -r requirements-yolo.txt` 로 설정합니다. 기존 서비스가 Blueprint와 연결된 경우에는 `FOOTFRAME_ENABLE_YOLO=1`로 설정하여 **재빌드**할 수도 있습니다.
3. Python 버전은 `PYTHON_VERSION=3.11.11`로 지정하고 배포합니다.
4. 사이트에서 `/arch-status`를 열어 `ready_to_attempt` 상태인지 확인합니다. 해당 상태는 **실제 추론 성공을 의미하지 않습니다.** 아치 사진을 한 장 올리고 결과 이미지가 나오는지 확인해야 합니다.

Render 기존 서비스의 수동 설정은 `render.yaml` 수정만으로 자동 변경되지 않을 수 있습니다. 모델을 실행할 수 없는 호스팅에서도 2D 컴퓨터비전 측정은 별도로 작동합니다.

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
3. Render 무료 서비스는 OpenCV 2D 측정용으로 Build Command `pip install -r requirements.txt`, Start Command `python app.py`, Health Check `/health`를 사용합니다.
4. YOLO까지 사용하려면 위의 별도 실행 환경과 설치 설정이 필요합니다. `render.yaml`의 `FOOTFRAME_ENABLE_YOLO=1` 옵션 또는 대시보드 Build Command를 사용하세요.

**개인 발 사진, 테스트 코드, 검증 파일은 공개 GitHub에 업로드하지 마세요.** `best.pt`는 첫 YOLO 분석 시 복원되는 파일이므로 `.gitignore`에서 제외되어 있습니다.


### GitHub 웹 업로드 실패 시

GitHub 브라우저는 한 파일당 25MiB까지만 업로드합니다. 기존 모델은 약 52MiB이므로 **`best.pt` 원본을 업로드하지 마세요.** 이 패키지에는 8MiB 이하의 `best.pt.part01`~`part07` 7개가 있고, 실행 시 `app.py`가 SHA256 확인 후 원본 모델을 자동 복원합니다.

- `.zip` 파일 자체를 GitHub에 올리지 말고 압축을 푼 파일들을 업로드합니다.
- **먼저** `app.py`, `README.md`, `requirements*.txt`, `render.yaml`, `.gitignore`를 저장소 루트에 올리고 커밋합니다.
- 다음으로 `models/arch/`의 7개 `best.pt.partNN` 및 `best.pt.sha256`을 폴더 구조를 유지하며 업로드합니다. 업로드 오류가 반복된다면 모델 조각은 2~3개씩 나누어 커밋합니다.
- `best.pt`(복원 모델), 다른 ZIP, 개인 촬영 사진은 올리지 않습니다.

## 검증 상태와 한계

기존 사진 `12290.jpg`~`12298.jpg` 9장으로 통합 전후 결과를 비교했으며 **발볼 95.83mm, 발길이 256.0mm**가 동일했습니다. 측면 6장 중 A4 후보는 6장, 개별 길이 계산은 5장, 최종값에 채택한 독립 검증 사진은 1장입니다. 이는 정확도 개선을 입증한 결과가 아닙니다. 기존 사용·제외 사진도 동일합니다. 제공하신 실측값 100mm·250mm와의 차이는 각각 **-4.17mm, +6.0mm**입니다. 실측값을 내부 보정 계수로 사용하지 않았습니다.

**이 결과는 잠정적인 A4 평면 투영 치수입니다.** 검은 칸의 바깥 사각형을 실제 A4 외곽으로 확정할 수 없는 사진이 있으며, 발의 높이·촬영 각도에 따른 시차와 다른 사람에 대한 측정 정확도도 검증되지 않았습니다. 신발 사이즈 확정값으로 사용하지 마세요.

## 실행 확인 범위

- 측면 실사진 6장 및 위쪽 3장 재분석: 사진별 측정값과 최종 채택 결과를 확인했습니다.
- `/health`, `/arch-status`, `/analyze`, `/analyze-arch`: 실제 HTTP 입력과 오류 응답을 검사했습니다.
- `predict_arch`의 BGR 전달·YOLO 세그멘테이션 결과 처리: 모의 모델로 경로를 테스트했습니다.
- YOLO `best.pt` 분할 파일은 SHA256으로 재결합 무결성을 확인했습니다.
- **실제 YOLO 추론은 이 실행 환경에 Ultralytics가 없고 설치 서버에도 접근할 수 없어 완료하지 못했습니다.** 모의 테스트를 실제 모델 추론으로 표시하지 않습니다.
- 실제 발 치수 정확도는 여전히 미검증입니다. 특히 측면의 A4 외곽 경계가 한 장에서만 독립 검증되었습니다.
