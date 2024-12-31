# Quantitative Investment Lab
 HGU 경영경제학부 퀀트투자랩
 - 지도교수: 김호현 교수님

## Research 1: 비트코인 수익률 예측 1차 연구
- Folder Name: btc_pred_rsch
- Date: 2023.12.27~

## Research 2: 비트코인 수익률 예측 2차 연구
- Folder Name: btc_pred_rsch_2
- Date: 2024.09.01~

Test Performance:
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

 Class0: Significant Decrease
 Class1: Insignificant Decrease
 Class2: Insignificant Increase
 Class3: Significant Increase
