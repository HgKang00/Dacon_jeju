import numpy as np


'''
# input
## train_df = training set
## test_df = test set
## not_col = data_frame 구성 시 빼고싶은 col name
## target = col 중 예측하고 싶은 것
'''
def standardization(train_df, test_df, not_col, target):
    train_df_ = train_df.copy()
    test_df_ = test_df.copy()
    col =  [col for col in list(train_df.columns) if col not in [not_col]]
    mean_list = []
    std_list = []
    for x in col:
        mean, std = train_df_.agg(["mean", "std"]).loc[:,x]
        mean_list.append(mean)
        std_list.append(std)
        train_df_.loc[:, x] = (train_df_[x] - mean) / std
        test_df_.loc[:, x] = (test_df_[x] - mean) / std
    return train_df_, test_df_, mean_list[col.index(target)], std_list[col.index(target)]

def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val

# 스케일링된 데이터를 역변환 함수 정의
def min_max_inverse_scaling(scaled_data, min_val, max_val):
    original_data = scaled_data * (max_val - min_val) + min_val
    return original_data


def z_score(data):
    mean_data = data.mean()
    std_data = data.std()
    scaled_data = (data - mean_data) / std_data
    return scaled_data, mean_data, std_data

def z_score_inverse(scaled_data, mean_data, std_data):
    original_data = scaled_data * std_data + mean_data
    return original_data
