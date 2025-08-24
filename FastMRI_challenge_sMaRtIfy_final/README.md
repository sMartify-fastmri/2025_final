# sMaRtIfy - Final Version

이 버전은 Team sMaRtIf$\mathbf{y}$가 재현상의 편의를 위해 제공하는 버전입니다. Version 2와 거의 모든 코드를 공유하지만 근본적인 설정이 달라 제출 모델에 대해 무작위성이 고정되지 않습니다. 학습 알고리즘상으로는 구현상의 이유로 hacker의 61-65에폭 learning rate가 조금 더 낮다는 것 이외에는 완전히 동일합니다. 디버깅을 충분히 하지 않은 버전이므로 사소한 오류가 있을 수 있습니다!

---

## 실행 가이드

### (0) 환경 설정

```bash
pip -r requirements.txt
```

### (1) Hacker 훈련

**Hacker**는 본 challenge의 모든 데이터에서 확인된 특수한 k-space mask의 패턴을 항상 적용하여 학습한 모델입니다. Hacker는 다음과 같이 재현할 수 있습니다.

```bash
sh hacker.sh
```

### (2) Helper 훈련

**Helper**는 공개된 데이터의 k-space mask 패턴을 벗어나는 비공개 데이터에 대응하는 강건한 모델입니다. Helper는 다음과 같이 재현할 수 있습니다.

1. Helper baseline을 45에폭 훈련합니다.
```bash
sh helper.sh
```
### (3) 두 모델 불러와서 합치기

1. Leaderboard 데이터의 분류에 사용될 최종 classifier를 훈련합니다.
    ```bash
    sh make_awesome_classifier.sh
    ```

2. 최종 expert model을 모두 모은 module을 만들어 reconstruct를 진행합니다.
    ```bash
    sh recon_and_eval.sh
    ``` 

---