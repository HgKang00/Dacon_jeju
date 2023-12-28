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
from DLinear import LTSF_DLinear
from DLinear import moving_avg
from NLinear import LTSF_NLinear
from jeju.utils.scalilng import standardization
from jeju.utils.scalilng import time_slide_df
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


data_list = data_split(data)


unique_code = ['BC_C_J', 'TG_B_J', 'CR_B_J', 'RD_E_S', 'BC_A_J', 'CB_F_J', 'RD_D_J', 'TG_A_S', 'BC_E_S', 'CR_D_J', 'BC_A_S', 'BC_B_S', 'TG_E_J', 
               'CR_E_S', 'RD_F_J', 'BC_E_J', 'TG_A_J', 'CR_C_J', 'CR_D_S', 'TG_C_J', 'CB_A_S', 'TG_D_J', 'CR_E_J', 'RD_C_S', 'BC_C_S', 'CB_E_J', 
               'RD_E_J', 'BC_D_J', 'CR_A_J', 'TG_E_S', 'TG_C_S', 'TG_D_S', 'RD_A_S', 'RD_A_J', 'RD_D_S', 'TG_B_S', 'CB_D_J', 'CB_A_J', 'BC_B_J']

## validation set for march
# march_data = {}
# for code in unique_code:
#     march_data[code] = {}
#     march_data[code]['2019'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2019-03-02') & (data_list[f'data_{code}']['timestamp'] <= '2019-03-26')].reset_index(drop = True)
#     march_data[code]['2020'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2020-03-07') & (data_list[f'data_{code}']['timestamp'] <= '2020-03-31')].reset_index(drop = True)
#     march_data[code]['2021'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2021-03-06') & (data_list[f'data_{code}']['timestamp'] <= '2021-03-30')].reset_index(drop = True)
#     march_data[code]['2022'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2022-03-05') & (data_list[f'data_{code}']['timestamp'] <= '2022-03-29')].reset_index(drop = True)
    
#     valid_avg = []
#     for i in range(25):
#         avg = (march_data[code]['2019']['price(원/kg)'][i] + march_data[code]['2020']['price(원/kg)'][i] + march_data[code]['2021']['price(원/kg)'][i] + march_data[code]['2022']['price(원/kg)'][i]) / 4
#         valid_avg.append(avg)
        
#     march_data[code]['avg'] = valid_avg

#     new_data = {
#         'ID': [f'{code}_20230304', f'{code}_20230305', f'{code}_20230306', f'{code}_20230307', f'{code}_20230308',
#             f'{code}_20230309', f'{code}_20230310', f'{code}_20230311', f'{code}_20230312', f'{code}_20230313',
#             f'{code}_20230314', f'{code}_20230315', f'{code}_20230316', f'{code}_20230317', f'{code}_20230318',
#             f'{code}_20230319', f'{code}_20230320', f'{code}_20230321', f'{code}_20230322', f'{code}_20230323',
#             f'{code}_20230324', f'{code}_20230325', f'{code}_20230326', f'{code}_20230327', f'{code}_20230328'],
#         'timestamp': pd.date_range(start='2023-03-04', end='2023-03-28'),
#         'item': [code.split('_')[0]] * 25,
#         'corporation': [code.split('_')[1]] * 25,
#         'location': [code.split('_')[2]] * 25,
#         'supply(kg)': [0.0] * 25,
#         'price(원/kg)': valid_avg
#     }

#     march_data[code]['valid_avg'] = pd.DataFrame(new_data)

march_data = {}
for code in unique_code:
    march_data[code] = {}
    march_data[code]['2019'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2019-02-22') & (data_list[f'data_{code}']['timestamp'] <= '2019-03-26')].reset_index(drop = True)
    march_data[code]['2020'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2020-02-29') & (data_list[f'data_{code}']['timestamp'] <= '2020-03-31')].reset_index(drop = True)
    march_data[code]['2021'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2021-02-27') & (data_list[f'data_{code}']['timestamp'] <= '2021-03-30')].reset_index(drop = True)
    march_data[code]['2022'] = data_list[f'data_{code}'][(data_list[f'data_{code}']['timestamp'] >= '2022-02-26') & (data_list[f'data_{code}']['timestamp'] <= '2022-03-29')].reset_index(drop = True)
    
    valid_avg = []
    for i in range(32):
        avg = (march_data[code]['2019']['price(원/kg)'][i] + march_data[code]['2020']['price(원/kg)'][i] + march_data[code]['2021']['price(원/kg)'][i] + march_data[code]['2022']['price(원/kg)'][i]) / 4
        valid_avg.append(avg)
        
    march_data[code]['avg'] = valid_avg

    new_data = {
        'ID': [f'{code}_20230225', f'{code}_20230226', 
            f'{code}_20230227', f'{code}_20230228', f'{code}_20230301', f'{code}_20230302', f'{code}_20230303',
            f'{code}_20230304', f'{code}_20230305', f'{code}_20230306', f'{code}_20230307', f'{code}_20230308',
            f'{code}_20230309', f'{code}_20230310', f'{code}_20230311', f'{code}_20230312', f'{code}_20230313',
            f'{code}_20230314', f'{code}_20230315', f'{code}_20230316', f'{code}_20230317', f'{code}_20230318',
            f'{code}_20230319', f'{code}_20230320', f'{code}_20230321', f'{code}_20230322', f'{code}_20230323',
            f'{code}_20230324', f'{code}_20230325', f'{code}_20230326', f'{code}_20230327', f'{code}_20230328'],
        'timestamp': pd.date_range(start='2023-02-25', end='2023-03-28'),
        'item': [code.split('_')[0]] * 32,
        'corporation': [code.split('_')[1]] * 32,
        'location': [code.split('_')[2]] * 32,
        'supply(kg)': [0.0] * 32,
        'price(원/kg)': valid_avg
    }

    march_data[code]['valid_avg'] = pd.DataFrame(new_data)




pred = {}

# for i in tqdm(range(len(unique_code))):
for i in tqdm(range(len(unique_code))):

    ## prepare dataset for training
    dataset_code = unique_code[i]

    train_df = data_list[f'data_{dataset_code}'].reset_index(drop=True)
    valid_df = march_data[dataset_code]['valid_avg']
    print(len(train_df))

    # train_df = train_df.drop(['ID', 'item', 'corporation', 'location'], axis=1)
    # test_df = test_df.drop(['ID', 'item', 'corporation', 'location'], axis=1)

    # scaling
    train_x = list(train_df['price(원/kg)'])
    valid_x = list(valid_df['price(원/kg)'])
    total_x = train_x + valid_x
    scaled_x, min_val, max_val = min_max_scaling(np.array(total_x))
    train_df['price(원/kg)'] = scaled_x[:1523]
    valid_df['price(원/kg)'] = scaled_x[1523:]
    # scaled_x, mean_data, std_data = z_score(train_df['price(원/kg)'])
    # train_df['price(원/kg)'] = scaled_x

    
    
    # 날짜 변환
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    valid_df['timestamp'] = pd.to_datetime(valid_df['timestamp'])
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
    

    train_x, train_y, train_date = time_slide_df(train_df, window_size, forcast_size, date, targets)
    valid_x, valid_y, valid_date = time_slide_df(valid_df, window_size, forcast_size, date, targets)


    train_ds = Data(train_x, train_y)
    valid_ds = Data(valid_x, valid_y)
    # test_ds = Data(test_x, test_y)

    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True,)
    valid_dl = DataLoader(valid_ds, batch_size = train_x.shape[0], shuffle=False)
    # test_dl  = DataLoader(test_ds,  batch_size = test_x.shape[0], shuffle=False)
    print("Success preprocessing")


    print("Paramater for training")
    print()
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    epochs = 100
    lr = 0.001
    # model paramater
    NLinear_model = LTSF_NLinear(
                                window_size = window_size, 
                                forcast_size = forcast_size, 
                                individual = True, 
                                feature_size = 1
                                )
    # Loss
    criterion = torch.nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(NLinear_model.parameters(), lr=lr)
    
    # early stopping
    early_stopping_epochs = 5
    best_loss = 99999
    early_stop_counter = 0

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

        if valid_loss > best_loss:
            early_stop_counter += 1
        else:
            best_loss = valid_loss
            early_stop_counter = 0

        # 조기 종료 조건 확인
        if early_stop_counter >= early_stopping_epochs:
            print(f"Early Stopping epoch:{epoch}")
            break
            
        data = train_y[-window_size:].reshape(1,-1)
        pred_value = np.array([])
        for i in range(28):
            pred_input = torch.Tensor(data).reshape(1,window_size,1)
            pred_output = NLinear_model(pred_input)

            data = np.append(data,pred_output.item())
            pred_value = np.append(pred_value,pred_output.item())
            data = data[1:]

        # pred[dataset_code] = pred_value
        # rescaling pred
        pred[dataset_code] = min_max_inverse_scaling(pred_value, min_val, max_val)

        # if epoch == epochs:
            # data = train_y[-window_size:].reshape(1,-1)
            # pred_value = np.array([])
            # for i in range(28):
            #     pred_input = torch.Tensor(data).reshape(1,window_size,1)
            #     pred_output = NLinear_model(pred_input)

            #     data = np.append(data,pred_output.item())
            #     pred_value = np.append(pred_value,pred_output.item())
            #     data = data[1:]

            # pred[dataset_code] = pred_value
        #     # rescaling pred
        #     pred[dataset_code] = min_max_inverse_scaling(pred_value, min_val, max_val)
        
        # print(pred_value)

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
        
        if epoch % 10 == 0:
            print("epoch = {}, train_loss : {:.3f}, valid_loss : {:.3f}".format(epoch, np.mean(loss_list), valid_loss))


# weights_list = {}
# weights_list['trend'] = NLinear_model.Linear_Trend.weight.detach().numpy()
# weights_list['seasonal'] = NLinear_model.Linear_Seasonal.weight.detach().numpy()

# for name, w in weights_list.items():    
#     fig, ax = plt.subplots()    
#     plt.title(name)
#     im = ax.imshow(w, cmap='plasma_r',)
#     fig.colorbar(im, pad=0.03)
#     plt.show()
#     plt.savefig("ex.png")



## Storing result

date_col = []
date = 20230304
for i in range(28):
    date_col.append(date + i)
# print(date_col)



result = []
for i in range(len(unique_code)):
    pred_adj = np.round(pred[unique_code[i]],1)

    for j in range(28):
        code = unique_code[i]
        final = pred_adj[j]
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

# 결과 데이터프레임 출력
# print(final_submission.head())


# 업데이트된 데이터프레임을 새로운 CSV 파일로 저장
final_submission.to_csv(f'csv/NLinear_{window_size}_{forcast_size}_minmax_validavg32.csv', index=False)