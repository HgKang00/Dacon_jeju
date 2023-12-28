import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def data_split(data):
    ID_split = data['ID'].str.split('_')
    code = []
    for l in ID_split:
        code.append(f"{l[0]}_{l[1]}_{l[2]}")
    unique_code = set(code)

    data_split_list = {}
    for code in unique_code:
        code_parts = code.split('_')
        code_item, code_corporation, code_location = code_parts[0], code_parts[1], code_parts[2]
        
        condition = (data['item'] == code_item) & (data['corporation'] == code_corporation) & (data['location'] == code_location)
        
        data_split_list[f"data_{code}"] = data[condition].reset_index(drop = True)

    return data_split_list

def time_slide_df(df, window_size, forcast_size, target): #date
    df_ = df.copy()
    data_list = []
    dap_list = []
    date_list = []
    for idx in range(0, df_.shape[0]-window_size-forcast_size+1):
        x = df_.loc[idx:idx+window_size-1, target].values.reshape(window_size, len(target))
        y = df_.loc[idx+window_size:idx+window_size+forcast_size-1, target].values
        # date_ = df_.loc[idx+window_size:idx+window_size+forcast_size-1, date].values
        data_list.append(x)
        dap_list.append(y)
        # date_list.append(date_)
    return np.array(data_list, dtype='float32'), np.array(dap_list, dtype='float32') #, np.array(date_list)


def date_range_to_numeric(start_date, end_date):
    # 시작 날짜와 종료 날짜를 datetime 객체로 변환
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    
    # 날짜 범위 계산
    date_list = []
    current_date = start_date_obj
    
    while current_date <= end_date_obj:
        date_list.append(int(current_date.strftime('%Y%m%d')))
        current_date += timedelta(days=1)
    
    return date_list
