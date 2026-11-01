
## 목적

T2 이미지를 사용하여 신생아의 MRI에서 MR로 보이는 뇌실주위 공간(PVS)을 자동으로 segmentation하는 정확도를 향상

## 방법

- 연구 대상: 출생 후 6일에서 100일 사이의 신생아 40명 (IRB 승인됨)
- 데이터셋: 32명 - 훈련용, 8명 - 검증용
- 모델: Swin UNETR 사용
  - T2 입력
  - SWI 입력
  - T2와 SWI 모두 입력
- 평가: Dice 유사 계수 (DSC) 사용

## 결과

- T2 입력: 평균 DSC = 0.871
- SWI 입력: 평균 DSC = 0.896
- T2와 SWI 모두 입력: 평균 DSC = 0.901

## 설치 방법

1. 저장소 클론:
    ```bash
    git clone https://github.com/changmaaa/BG_segmentation.git
    cd BG_segmentation
    ```

2. 가상 환경 생성 및 패키지 설치:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
