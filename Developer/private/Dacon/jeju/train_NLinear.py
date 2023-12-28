## 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

##
from model import LTSF_NLinear
from jeju.utils.scalilng import standardization
from jeju.utils.scalilng import Data
from jeju.utils.scalilng import data_split
from jeju.utils.scalilng import date_range_to_numeric
from jeju.utils.scalilng import min_max_inverse_scaling
from jeju.utils.scalilng import min_max_scaling
from jeju.utils.scalilng import z_score
from jeju.utils.scalilng import z_score_inverse



# data_loading
data = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 상품, 회사, 지역 별로 분할하여 df_list에 저장
# unique code = {'CB_E_J', 'TG_A_S', 'TG_D_S', 'CB_A_S', 'RD_C_S', 'BC_E_J', 'CR_D_S', 'BC_C_J', 'RD_D_S', 'BC_A_J', 'BC_C_S', 'TG_C_J', 'TG_D_J', 
#                'TG_C_S', 'BC_D_J', 'TG_E_S', 'RD_D_J', 'CR_B_J', 'RD_A_J', 'TG_B_J', 'BC_A_S', 'CR_A_J', 'TG_B_S', 'CR_E_S', 'TG_E_J', 'CR_D_J', 
#                'TG_A_J', 'RD_E_J', 'BC_B_S', 'RD_F_J', 'CR_E_J', 'CB_A_J', 'BC_B_J', 'CR_C_J', 'CB_F_J', 'RD_E_S', 'RD_A_S', 'CB_D_J', 'BC_E_S'}

data_list = data_split(data)
test_list = data_split(test)

unique_code = ['BC_C_J', 'TG_B_J', 'CR_B_J', 'RD_E_S', 'BC_A_J', 'CB_F_J', 'RD_D_J', 'TG_A_S', 'BC_E_S', 'CR_D_J', 'BC_A_S', 'BC_B_S', 'TG_E_J', 
               'CR_E_S', 'RD_F_J', 'BC_E_J', 'TG_A_J', 'CR_C_J', 'CR_D_S', 'TG_C_J', 'CB_A_S', 'TG_D_J', 'CR_E_J', 'RD_C_S', 'BC_C_S', 'CB_E_J', 
               'RD_E_J', 'BC_D_J', 'CR_A_J', 'TG_E_S', 'TG_C_S', 'TG_D_S', 'RD_A_S', 'RD_A_J', 'RD_D_S', 'TG_B_S', 'CB_D_J', 'CB_A_J', 'BC_B_J']

def time_slide_df(df, window_size, forcast_size, date, target):
    df_ = df.copy()
    data_list = []
    dap_list = []
    date_list = []
    for idx in range(0, df_.shape[0]-window_size-forcast_size+1):
        x = df_.loc[idx:idx+window_size-1, target].values.reshape(window_size, 1)
        y = df_.loc[idx+window_size:idx+window_size+forcast_size-1, target].values
        date_ = df_.loc[idx+window_size:idx+window_size+forcast_size-1, date].values
        data_list.append(x)
        dap_list.append(y)
        date_list.append(date_)
    return np.array(data_list, dtype='float32'), np.array(dap_list, dtype='float32'), np.array(date_list)






pred = {}

# for i in tqdm(range(len(unique_code))):
for i in tqdm(range(len(unique_code))):

    ## prepare dataset for training
    dataset_code = unique_code[i]

    train_df = data_list[f'data_{dataset_code}'].reset_index(drop=True)
    test_df = test_list[f'data_{dataset_code}'].reset_index(drop=True)
    print(len(train_df))

    # train_df = train_df.drop(['ID', 'item', 'corporation', 'location'], axis=1)
    # test_df = test_df.drop(['ID', 'item', 'corporation', 'location'], axis=1)

    # scaling
    scaled_x, min_val, max_val = min_max_scaling(train_df['price(원/kg)'])
    train_df['price(원/kg)'] = scaled_x
    # scaled_x, mean_data, std_data = z_score(train_df['price(원/kg)'])
    # train_df['price(원/kg)'] = scaled_x

    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    print("Dataset prepared")
    print()

    ## paramaters
    window_size = 7
    forcast_size= 1
    batch_size = 32
    targets = 'price(원/kg)'
    date = 'timestamp'
    not_col = 'timestamp'

    print("Start preprocessing")
    print()
    # train_df_fe, test_df_fe, mean_, std_ = standardization(train_df, test_df, not_col, targets)
    # train_x, train_y, train_date = time_slide_df(train_df_fe, window_size, forcast_size, date, targets)
    # test_x, test_y, test_date = time_slide_df(test_df_fe, window_size, forcast_size, date, targets)

    train_x, train_y, train_date = time_slide_df(train_df, window_size, forcast_size, date, targets)
    # test_x, test_y, test_date = time_slide_df(test_df, window_size, forcast_size, date, targets)


    train_ds = Data(train_x[:1400], train_y[:1400])
    valid_ds = Data(train_x[1400:], train_y[1400:])
    # test_ds = Data(test_x, test_y)

    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True,)
    valid_dl = DataLoader(valid_ds, batch_size = train_x[1400:].shape[0], shuffle=False)
    # test_dl  = DataLoader(test_ds,  batch_size = test_x.shape[0], shuffle=False)
    print("Success preprocessing")


    print("Paramater for training")
    print()
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    epochs = 60
    lr = 0.001
    NLinear_model = LTSF_NLinear(
                                window_size = window_size, 
                                forcast_size = forcast_size, 
                                individual = True, 
                                feature_size = 1
                                )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(NLinear_model.parameters(), lr=lr)
    max_loss = 999999999

    print("strat training")
    print()


    for epoch in tqdm(range(1, epochs+1)):
        loss_list = []
        NLinear_model.train()
        for batch_eeeeidx, (data, target) in enumerate(train_dl):
            optimizer.zero_grad()
            output = NLinear_model(data)
            loss = criterion(output, target.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            loss_list.append(np.sqrt(loss.item()))  
        train_loss_list.append(np.sqrt(np.mean(loss_list)))

        NLinear_model.eval()
        with torch.no_grad():
            for data, target in valid_dl:
                output = NLinear_model(data)
                valid_loss = np.sqrt(criterion(output, target.unsqueeze(-1)).item())
                valid_loss_list.append(np.sqrt(valid_loss.item()))
            
        if epoch == epochs:
            data = train_y[-7:].reshape(1,-1)
            pred_value = np.array([])
            for i in range(28):
                pred_input = torch.Tensor(data).reshape(1,7,1)
                pred_output = NLinear_model(pred_input)
                data = np.append(data,pred_output.item())
                pred_value = np.append(pred_value,pred_output.item())
                data = data[1:]

            # pred[dataset_code] = pred_value
            # rescaling pred
            pred[dataset_code] = min_max_inverse_scaling(pred_value, min_val, max_val)
            # pred[dataset_code] = z_score_inverse(pred_value, mean_data, std_data)

            # for data, target in test_dl:
            #     output = NLinear_model(data)
            #     test_loss = criterion(output, target.unsqueeze(-1))
            #     test_loss_list.append(test_loss)

        # if valid_loss < max_loss:
        #     torch.save(NLinear_model, f'NLinear_model_{dataset_code}.pth')
        #     max_loss = valid_loss
        #     # print("valid_loss={:.3f}, test_los{:.3f}, Model Save".format(valid_loss, test_loss))
        #     print("valid_loss={:.3f}, Model Save".format(valid_loss))
        #     dlinear_best_epoch = epoch
        #     dlinear_best_train_loss = np.mean(loss_list)
        #     dlinear_best_valid_loss = np.mean(valid_loss.item())

        # print("epoch = {}, train_loss : {:.3f}, valid_loss : {:.3f}, test_loss : {:.3f}".format(epoch, np.mean(loss_list), valid_loss, test_loss))
        
        if epoch % 20 == 0:
            print("epoch = {}, train_loss : {:.3f}, valid_loss : {:.3f}".format(epoch, np.mean(loss_list), valid_loss))



# ## 후보정
# march_data = {}
# for code in unique_code:
#     march_data[code] = {}
#     march_data[code]['2019'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2019-02-23') & (data_list[f'data_{code}']['timestamp'] <= '2019-03-29')].reset_index(drop = True)
#     march_data[code]['2020'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2020-02-29') & (data_list[f'data_{code}']['timestamp'] <= '2020-04-03')].reset_index(drop = True)
#     march_data[code]['2021'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2021-02-27') & (data_list[f'data_{code}']['timestamp'] <= '2021-04-02')].reset_index(drop = True)
#     march_data[code]['2022'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2022-02-26') & (data_list[f'data_{code}']['timestamp'] <= '2022-04-01')].reset_index(drop = True)
    
#     valid_avg = []
#     for i in range(35):
#         price_19 = march_data[code]['2019']['price(원/kg)'][i]
#         price_20 = march_data[code]['2020']['price(원/kg)'][i] 
#         price_21 = march_data[code]['2021']['price(원/kg)'][i] 
#         price_22 =march_data[code]['2022']['price(원/kg)'][i] 

#         prices = [price_19, price_20, price_21, price_22]
#         filtered_prices = [price for price in prices if price != 0]

#         # 평균 계산
#         if filtered_prices:
#             average_price = sum(filtered_prices) / len(filtered_prices)
#         else: 
#             average_price = 0
        
#         valid_avg.append(average_price)
        
#     march_data[code]['avg'] = valid_avg

#     new_data = {
#         'ID': [f'{code}_20230225', f'{code}_20230226', 
#             f'{code}_20230227', f'{code}_20230228', f'{code}_20230301', f'{code}_20230302', f'{code}_20230303',
#             f'{code}_20230304', f'{code}_20230305', f'{code}_20230306', f'{code}_20230307', f'{code}_20230308',
#             f'{code}_20230309', f'{code}_20230310', f'{code}_20230311', f'{code}_20230312', f'{code}_20230313',
#             f'{code}_20230314', f'{code}_20230315', f'{code}_20230316', f'{code}_20230317', f'{code}_20230318',
#             f'{code}_20230319', f'{code}_20230320', f'{code}_20230321', f'{code}_20230322', f'{code}_20230323',
#             f'{code}_20230324', f'{code}_20230325', f'{code}_20230326', f'{code}_20230327', f'{code}_20230328',
#             f'{code}_20230329',f'{code}_20230330',f'{code}_20230331'],
#         'timestamp': pd.date_range(start='2023-02-25', end='2023-03-31'),
#         'item': [code.split('_')[0]] * 35,
#         'corporation': [code.split('_')[1]] * 35,
#         'location': [code.split('_')[2]] * 35,
#         'supply(kg)': [0.0] * 35,
#         'price(원/kg)': valid_avg
#     }

#     march_data[code]['valid_avg'] = pd.DataFrame(new_data)

## Storing result

date_col = []
date = 20230304
for i in range(28):
    date_col.append(date + i)
# print(date_col)


result = []
for i in range(len(unique_code)):
    pred_adj = np.round(pred[unique_code[i]],1)
    pred_true = np.round(march_data[unique_code[i]]['avg'][7:],1)


    for j in range(28):
        code = unique_code[i]
        final = (pred_adj[j] + pred_true[j]) / 2
        if date_col[j] in [20230305, 20230312, 20230319, 20230326]:
            final = 0
        else:
            final = final

        code_time = f"{unique_code[i]}_{date_col[j]}"
        result.append([code_time, final])
    
    
result_df = pd.DataFrame(result, columns = ['ID', 'pred'])
submission = pd.read_csv("data/sample_submission.csv")


# 'submission' 데이터프레임과 'result_df' 데이터프레임을 'ID'를 기준으로 병합
final_submission = submission.merge(result_df, on='ID', how='left')

# 'pred' 값을 'answer' 열에 복사
final_submission['answer'] = final_submission['pred']

# 'pred' 열 삭제
final_submission.drop('pred', axis=1, inplace=True)

# 60보다 작은 경우 0으로 대체
final_submission['answer'] = final_submission['answer'].apply(lambda x: 0 if x < 60 else x)

# 결과 데이터프레임 출력
print(final_submission.head())


# 업데이트된 데이터프레임을 새로운 CSV 파일로 저장
final_submission.to_csv(f'csv/NLinear_{window_size}_{forcast_size}_epochs{epochs}_postprocess_minmax.csv', index=False)