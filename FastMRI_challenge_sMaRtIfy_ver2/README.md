# sMaRtIfy - Version 2

이 버전은 실제로 Team sMaRtIf$\mathbf{y}$가 제출용 모델을 훈련하기 위해 사용한 버전입니다. 최대한의 재현성을 확보하면서 동일 모델을 학습하고 싶으시다면 본 버전을 사용해주세요.

---

## 실행 가이드

### (0) 환경 설정

1. 코드 실행에 필요한 라이브러리를 설치합니다.
    ```bash
    pip -r requirements.txt
    ```

### (1) Hacker 훈련

**Hacker**는 본 challenge의 모든 데이터에서 확인된 특수한 k-space mask의 패턴을 항상 적용하여 학습한 모델입니다. Hacker는 다음과 같이 재현할 수 있습니다.

1. Hacker baseline을 45에폭 훈련합니다.
    ```bash
    sh hacker_baseline.sh
    ```
    향상된 재현성을 원하는 경우, 19에폭(20번째 에폭)까지 훈련된 모델을 불러와 20에폭부터 다시 학습합니다. 실행 결과는 ```./experiments/terminal_log/```의 ```[IABENG41] hacker_baseline.txt```에서 비교해볼 수 있습니다.
    $\tiny\textcolor{red}{\text{* 종종 알 수 없는 이유로 ../Data\_mask/ 내부의 numpy 파일이 손상됩니다. 이 경우 아래 명령을 실행하면 문제없이 훈련이 재개됩니다.}}$
    ```bash
    sh hacker_baseline_contd.sh
    ```

2. 위의 baseline 모델을 30에폭에서 불러와 MOE와 MRAugment를 적용합니다. 실행 결과는 ```./experiments/terminal_log/```의 ```[IABENG41] hacker_moe.txt```에서 비교해볼 수 있습니다.
    ```bash
    sh hacker_moe.sh
    ```

3. 위의 MOE 모델을 불러와 5에폭 추가로 훈련하는 다음 shell 파일을 실행해주세요.
    $\tiny\textcolor{red}{\text{* 구현상의 이유로 30에폭부터 시작하는 듯한 UI를 출력하나, 무시하고 실행해주세요.}}$
    ```bash
    sh hacker_moe_clutch.sh
    ```

### (2) Helper 훈련

**Helper**는 공개된 데이터의 k-space mask 패턴을 벗어나는 비공개 데이터에 대응하는 강건한 모델입니다. Helper는 다음과 같이 재현할 수 있습니다.

1. Helper baseline을 45에폭 훈련합니다.
    ```bash
    sh helper_baseline.sh
    ```
    향상된 재현성을 원하는 경우, 해당 모델이 45에폭에 도달하여 훈련이 끝났을 때 16에폭까지 훈련된 모델을 불러와 17에폭부터 다시 학습합니다. 실행 결과는 ```./experiments/terminal_log/```의 ```[IABENG95] helper_baseline.txt```에서 비교해볼 수 있습니다.
    $\tiny\textcolor{red}{\text{* Helper 학습 후 코드 실수를 발견하여, 해당 실수의 영향이 없는 16에폭까지의 모델을 불러와 다시 학습했습니다. 16에폭 이전의 seed와는 무관합니다.}}$
    ```bash
    sh helper_baseline_contd.sh
    ```

2. 위의 baseline 모델을 30에폭에서 불러와, brain과 knee 각각 다른 모델을 만들어 MOE와 MRAugment를 적용합니다. 실행 결과는 ```./experiments/terminal_log/```의 ```[IABENG95] helper_moe_rescue_brain.txt```와 ```[IABENG71] helper_moe_rescue_knee.txt```에서 비교해볼 수 있습니다.
    $\tiny\textcolor{red}{\text{* 위에서 언급한 코드 실수로 학습 시간이 부족해져, 두 서버에서 brain과 knee 모델을 각각 나누어 훈련했습니다.}}$
    ```bash
    sh helper_moe_rescue_brain.sh
    sh helper_moe_rescue_knee.sh
    ```

### (3) 두 모델 불러와서 합치기

1. Leaderboard 데이터의 분류에 사용될 최종 classifier를 훈련합니다.
    ```bash
    sh make_awesome_classifier.sh
    ```
2. 최종 expert model을 모두 모은 module을 저장합니다.
    ```bash
    sh collect_exodia.sh
    ``` 
    Hacker의 경우 평가 지표가 존재하므로, 위의 명령을 실행하기 **전에** 아래 명령을 통해 공개 리더보드에서 가장 좋은 성능을 보여준 expert를 선택해 저장합니다. 제출된 모델은 **이미** 평가되어 위 스크립트에 반영되어 있습니다.
    ```bash
    bash recon_eval_all_hacker.sh
    ``` 
    대회 규칙상 재현 실험에서 이를 적용하기 어려울 경우, 위의 두 명령 대신 최종 에폭 모델을 불러오는 `collect_last.sh`를 실행합니다. 대부분의 경우, 모든 expert model에서 최종 에폭 모델이 가장 좋습니다.
    ```bash
    sh collect_last.sh
    ``` 

3. 최종 module을 이용해 평가를 진행합니다.
    ```bash
    sh reconstruct_final.sh
    sh leaderboard_eval_final.sh
    ```

---