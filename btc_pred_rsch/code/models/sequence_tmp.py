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


from tqdm import tqdm

def createSeqForBacktestwithTimeGapDelete(df, seq_len):
    """
    sequence 생성 시, num_rows가 5이하 이거나 시간 공백이 존재하면 그 sequence 제외하는 함수
    """
    # window_start 컬럼을 datetime 유형으로 변환
    df['window_start'] = pd.to_datetime(df['window_start'])

    # 종속변수 및 필요 변수 설정
    target_var_lst = ['returns', 'returns_next10m', 'realized_vol_next10m']
    target_var = 'returns_next10m'  # 종속변수

    # 필요없는 컬럼 삭제
    df.drop(columns=['window_end', 'num_rows', 'time_id'], inplace=True)

    # target을 제외한 나머지 종속변수 삭제
    cols_to_drop = [var for var in target_var_lst if var != target_var]
    df.drop(columns=cols_to_drop, inplace=True)

    # 종속변수를 데이터 프레임 맨 뒤로 옮기기
    cols = [col for col in df.columns if col not in [target_var, 'returns_next10m_binary']] + [target_var, 'returns_next10m_binary']
    df = df[cols]

    # 시간 간격 계산
    df['time_gap'] = df['window_start'].diff().dt.total_seconds().gt(30).cumsum()
    
    # sequence 생성
    sequences = []
    scaler = MinMaxScaler()
    
    grouped = df.groupby('time_gap')
    for name, group in tqdm(grouped):
        if group['del_idx'].sum() > 0:
            continue

        for start_idx in range(len(group) - seq_len + 1):
            sequence = group.iloc[start_idx:start_idx + seq_len]
            
            # 스케일링 및 시퀀스 준비
            scaled_sequence = scaler.fit_transform(sequence.drop(columns=['del_idx', 'window_start', target_var, 'returns_next10m_binary', 'time_gap']))
            scaled_sequence_with_target = pd.concat([pd.DataFrame(scaled_sequence), sequence[[target_var, 'returns_next10m_binary']].reset_index(drop=True)], axis=1)
            sequences.append(scaled_sequence_with_target.values)

    sequences = np.array(sequences)

    # X, y, y_for_backtest 분리
    X = sequences[:, :, :-2]
    y = sequences[:, -1, -1].reshape(-1, 1)
    y_for_backtest = sequences[:, -1, -2:].reshape(-1, 2)

    return X, y, y_for_backtest


def createSeqForBacktestwithTimeGapDelete_tmp(df, seq_len):
    """
    sequence 생성 시, num_rows가 5이하 이거나 시간 공백이 존재하면 그 sequence 제외하는 함수
    이 함수는 디버깅을 위한 함수임. window_start 값을 반환함으로써 작업이 잘 수행됐는지 확인하기 위한 함수
    """
    # window_start 컬럼을 datetime 유형으로 변환
    df['window_start'] = pd.to_datetime(df['window_start'])

    # 종속변수 및 필요 변수 설정
    target_var_lst = ['returns', 'returns_next10m', 'realized_vol_next10m']
    target_var = 'returns_next10m'  # 종속변수

    # 필요없는 컬럼 삭제
    df.drop(columns=['num_rows', 'time_id'], inplace=True)

    # target을 제외한 나머지 종속변수 삭제
    cols_to_drop = [var for var in target_var_lst if var != target_var]
    df.drop(columns=cols_to_drop, inplace=True)

    # 종속변수를 데이터 프레임 맨 뒤로 옮기기
    cols = [col for col in df.columns if col not in [target_var, 'returns_next10m_binary']] + [target_var, 'returns_next10m_binary']
    df = df[cols]

    # 시간 간격 계산
    df['time_gap'] = df['window_start'].diff().dt.total_seconds().gt(30).cumsum()
    
    # sequence 생성
    sequences = []
    scaler = MinMaxScaler()
    
    grouped = df.groupby('time_gap')
    for name, group in tqdm(grouped):
        if group['del_idx'].sum() > 0:
            continue

        for start_idx in range(len(group) - seq_len + 1):
            sequence = group.iloc[start_idx:start_idx + seq_len]
            
            # 스케일링할 컬럼 선택
            #scale_columns = [col for col in sequence.columns if col not in ['del_idx', 'window_start', target_var, 'returns_next10m_binary', 'time_gap']]
            scale_columns = ['lowest_return', 'highest_return',
       'high_low_gap', 'trade_vol', 'volume_power', 'beginning_price',
       'ending_price', 'lowest_price', 'highest_price', 'returns_next10m',
       'ob_end_ap_0', 'ob_end_as_0', 'ob_end_bp_0', 'ob_end_bs_0',
       'ob_end_ap_1', 'ob_end_as_1', 'ob_end_bp_1', 'ob_end_bs_1',
       'ob_end_ap_2', 'ob_end_as_2', 'ob_end_bp_2', 'ob_end_bs_2',
       'ob_end_ap_3', 'ob_end_as_3', 'ob_end_bp_3', 'ob_end_bs_3',
       'ob_end_ap_4', 'ob_end_as_4', 'ob_end_bp_4', 'ob_end_bs_4',
       'ob_end_ap_5', 'ob_end_as_5', 'ob_end_bp_5', 'ob_end_bs_5',
       'ob_end_ap_6', 'ob_end_as_6', 'ob_end_bp_6', 'ob_end_bs_6',
       'ob_end_ap_7', 'ob_end_as_7', 'ob_end_bp_7', 'ob_end_bs_7',
       'ob_end_ap_8', 'ob_end_as_8', 'ob_end_bp_8', 'ob_end_bs_8',
       'ob_end_ap_9', 'ob_end_as_9', 'ob_end_bp_9', 'ob_end_bs_9',
       'ob_end_ap_10', 'ob_end_as_10', 'ob_end_bp_10', 'ob_end_bs_10',
       'ob_end_ap_11', 'ob_end_as_11', 'ob_end_bp_11', 'ob_end_bs_11',
       'ob_end_ap_12', 'ob_end_as_12', 'ob_end_bp_12', 'ob_end_bs_12',
       'ob_end_ap_13', 'ob_end_as_13', 'ob_end_bp_13', 'ob_end_bs_13',
       'ob_end_ap_14', 'ob_end_as_14', 'ob_end_bp_14', 'ob_end_bs_14',
       'ob_end_bias_0', 'ob_end_bias_1', 'ob_end_bias_4',
       'ob_end_bidask_spread', 'ob_end_liq_0', 'ob_end_liq_1', 'ob_end_liq_4',
       'highest_possible_return']
            scaled_data = scaler.fit_transform(sequence[scale_columns])
            
            # 스케일링 및 시퀀스 준비
            scaled_sequence_with_target = pd.concat([sequence[['window_start']].reset_index(drop=True), pd.DataFrame(scaled_data, columns=scale_columns), sequence[[target_var, 'returns_next10m_binary']].reset_index(drop=True)], axis=1)
            sequences.append(scaled_sequence_with_target.values)

    sequences = np.array(sequences)

    # X, y, y_for_backtest 분리
    X = sequences[:, :, 1:-2]  # window_start 컬럼을 제외
    y = sequences[:, -1, -1].reshape(-1, 1)
    y_for_backtest = sequences[:, -1, -2:].reshape(-1, 2)
    df_tmp = pd.DataFrame(sequences[-1], columns=['window_start'] + [f'feature_{i}' for i in range(sequences.shape[2]-3)] + ['returns_next10m', 'returns_next10m_binary'])  # 마지막 시퀀스를 데이터프레임으로 변환


    return X, y, y_for_backtest, df_tmp

