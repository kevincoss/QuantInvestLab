# Quantitative Investment Lab
 HGU 경영경제학부 퀀트투자랩
 - 지도교수: 김호현 교수님

## Research 1: 비트코인 수익률 예측 연구
- Folder Name: btc_pred_rsch
- Date: 2023.12.27~

## Research 2: 비트코인 수익률 예측 연구 (2차 연구)
- Folder Name: btc_pred_rsch_2
- Date: 2024.09.01~

### 프로젝트 개요
이 프로젝트는 초단기 비트코인 데이터(0.01초 단위)를 활용하여 유의미한 상승/하락 및 유의미하지 않은 상승/하락을 예측하는 4중 분류 문제를 해결하기 위해 진행되었습니다. 다양한 차트 지표를 독립변수로 활용하며, 딥러닝 및 머신러닝 모델을 사용하여 수익률 예측의 정확도를 높이는 것을 목표로 합니다.

### 연구 목적
1. 비트코인 초단기 변동성 패턴 분석: 시장의 움직임을 세밀하게 파악하여 투자 전략에 활용할 수 있는 인사이트 제공.
2. 4중 분류 모델 개발: 유의미한 상승, 유의미한 하락, 유의미하지 않은 상승, 유의미하지 않은 하락을 정확히 분류.
3. 독립변수의 효과 분석: 주요 차트 지표와 가격 데이터의 상관관계를 탐구하여 예측 성능을 최적화.

### 데이터 설명
- 데이터 기간: 0.01초 단위 비트코인 거래 데이터
- 종속변수 (Target):
 - 유의미한 상승
 - 유의미한 하락
 - 유의미하지 않은 상승
 - 유의미하지 않은 하락

- 독립변수 (Features): 총 80개 이상의 차트 지표를 사용하여 모델을 훈련.
 - 주요 변수:
  - ask_price_n 및 bid_price_n: 매도/매수 가격 (1~15차 레벨)
  - ask_size_n 및 bid_size_n: 매도/매수 사이즈 (1~15차 레벨)
  - total_ask_size, total_bid_size: 총 매도/매수 사이즈
  - latest_trade_price, latest_log_return: 최근 거래 가격 및 로그 수익률
  - EMA4, EMA9, EMA18: 지수 이동 평균 (Exponential Moving Average)
  - RSI3: 상대 강도 지수 (Relative Strength Index)
  - bid_ask_spread, mid_price, mid_price_diff: 매도-매수 스프레드 및 중간 가격
  - 1_min_return, 5_min_return: 1분/5분 단위 수익률
  - slope1 ~ slope7: 기울기 지표
  - 기타 파생 변수

### 결과

              precision    recall  f1-score   support
           0     0.1410    0.0151    0.0272      7972
           1     0.3451    0.0610    0.1037     20119
           2     0.3642    0.9262    0.5229     22326
           3     0.5987    0.0666    0.1199     12114

    accuracy                         0.3652     62531
   macro avg     0.3623    0.2672    0.1934     62531
weighted avg     0.3750    0.3652    0.2467     62531

![image](https://github.com/user-attachments/assets/3d60cd7f-0d36-4e26-9915-48a33a240021)

Best model + Best Preprocessing 기준 
예측하고자 하는 Significant Increase (0.02% 이상의 수익)의 Precision 기준 0.6퍼센트 정확도 예측 성능

- Class0: Significant Decrease
- Class1: Insignificant Decrease
- Class2: Insignificant Increase
- Class3: Significant Increase
