"""
Note: bitcoin_prepro_v2.ipynb 파일을 통해 생성된 df에 적용할 것
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequence(df, seq_len):
    # 1. 변수 선택
    # 종속변수 리스트
    target_var_lst = ['returns', 'returns_next10m', 'realized_vol_next10m']
    target_var = 'returns_next10m' # 종속변수

    # 시퀀스 생성 전 필요없는 컬럼 삭제
    df.drop(columns=['window_start', 'window_end','num_rows', 'time_id'], inplace=True) # 시간 관련 변수

    # target을 제외한 나머지 종속변수 삭제
    cols_to_drop = [var for var in target_var_lst if var != target_var]
    df.drop(columns=cols_to_drop, inplace=True) # 종속변수
    #df['returns_next10m'] = df['returns_next10m'].apply(lambda x: 0 if x <= 0 else 1) 2진 분류화는 ipynb 파일에서 직접 수행

    # 종속변수를 데이터 프레임 맨 뒤로 옮기기
    cols = df.columns.tolist()
    cols = [col for col in cols if col != 'returns_next10m'] + ['returns_next10m'] # 종속변수 맨 뒤로
    df = df[cols]

    # 2. sequence 생성
    sequences = []
    scaler = MinMaxScaler()
    
    for start_idx in range(len(df) - seq_len + 1):  # 데이터 프레임을 순회하며 시퀀스 생성
        end_idx = start_idx + seq_len
        sequence = df.iloc[start_idx:end_idx]
        
        # 시퀀스 내에 del_idx가 1인 행이 있다면, 해당 시퀀스를 제외
        if sequence['del_idx'].sum() == 0:
            # 예측하고자 하는 마지막 피처의 값을 제외하고 스케일링
            scaled_sequence = scaler.fit_transform(sequence.drop(columns=['del_idx', target_var]))
            
            # 스케일링된 시퀀스에 예측하고자 하는 마지막 피처의 값을 추가
            scaled_sequence_with_target = pd.concat([pd.DataFrame(scaled_sequence), sequence[target_var].reset_index(drop=True)], axis=1)
            
            # 최종 시퀀스 추가
            sequences.append(scaled_sequence_with_target.values)
            
    # 3. X, y split
    sequences = np.array(sequences)
    # X와 y를 분리
    X = sequences[:, :, :-1] # 마지s막 시퀀스와 마지막 컬럼을 제외한 나머지
    y = sequences[:, -1, -1].reshape(-1, 1) # 각 시퀀스의 마지막 행, 마지막 컬럼의 값

    return X, y



def createSeqForBacktest(df, seq_len):
    # 1. 변수 선택
    target_var_lst = ['returns', 'returns_next10m', 'realized_vol_next10m']
    target_var = 'returns_next10m' # 종속변수

    # 시퀀스 생성 전 필요없는 컬럼 삭제
    df.drop(columns=['window_start', 'window_end','num_rows', 'time_id'], inplace=True)

    # target을 제외한 나머지 종속변수 삭제
    cols_to_drop = [var for var in target_var_lst if var != target_var]
    df.drop(columns=cols_to_drop, inplace=True)

    # 종속변수를 데이터 프레임 맨 뒤로 옮기기
    cols = [col for col in df.columns if col not in ['returns_next10m', 'returns_next10m_binary']] + ['returns_next10m', 'returns_next10m_binary']
    df = df[cols]

    # 2. sequence 생성
    sequences = []
    scaler = MinMaxScaler()
    
    for start_idx in range(len(df) - seq_len + 1):
        end_idx = start_idx + seq_len
        sequence = df.iloc[start_idx:end_idx]
        
        if sequence['del_idx'].sum() == 0:
            scaled_sequence = scaler.fit_transform(sequence.drop(columns=['del_idx', 'returns_next10m', 'returns_next10m_binary']))
            scaled_sequence_with_target = pd.concat([pd.DataFrame(scaled_sequence), sequence[['returns_next10m', 'returns_next10m_binary']].reset_index(drop=True)], axis=1)
            sequences.append(scaled_sequence_with_target.values)
            
    sequences = np.array(sequences)

    # 3. X, y, y_for_backtest split
    X = sequences[:, :, :-2] # 마지막 두 컬럼을 제외한 나머지
    y = sequences[:, -1, -1].reshape(-1, 1) # 각 시퀀스의 마지막 행, 마지막 컬럼
    y_for_backtest = sequences[:, -1, -2:].reshape(-1, 2) # 각 시퀀스의 마지막 행, 끝에서 두 번째와 마지막 컬럼

    return X, y, y_for_backtest

