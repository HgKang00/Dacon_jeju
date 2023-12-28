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
from jeju.utils.scalilng import standardization
from jeju.utils.scalilng import time_slide_df
from jeju.utils.scalilng import Data
from jeju.utils.scalilng import data_split
from jeju.utils.scalilng import date_range_to_numeric
from jeju.utils.scalilng import min_max_inverse_scaling
from jeju.utils.scalilng import min_max_scaling
from jeju.utils.scalilng import z_score
from jeju.utils.scalilng import z_score_inverse
from DLinear_addsupply import LTSF_DLinear_addsupply


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
    scaled_x1, min_val1, max_val1 = min_max_scaling(train_df['supply(kg)'])
    train_df['supply(kg)'] = scaled_x1

    # scaled_x, mean_data, std_data = z_score(train_df['price(원/kg)'])
    # train_df['price(원/kg)'] = scaled_x

    
    
    # 날짜 변환
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    print("Dataset prepared")
    print()

    ## paramaters
    window_size = 7
    forcast_size= 1
    batch_size = 32
    targets = ['supply(kg)','price(원/kg)']
    date = 'timestamp'
    not_col = 'timestamp'

    print("Start preprocessing")
    print()
    

    train_x, train_y, train_date = time_slide_df(train_df, window_size, forcast_size, date, targets)
    


    train_ds = Data(train_x[:1400], train_y[:1400])
    valid_ds = Data(train_x[1400:], train_y[1400:])
    # test_ds = Data(test_x, test_y)

    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=False,)
    valid_dl = DataLoader(valid_ds, batch_size = train_x[1400:].shape[0], shuffle=False)
    # test_dl  = DataLoader(test_ds,  batch_size = test_x.shape[0], shuffle=False)
    print("Success preprocessing")


    print("Paramater for training")
    print()
    train_loss_multi_list = []
    train_loss_list = []

    valid_loss_multi_list = []
    valid_loss_list = []

    epochs = 100
    lr = 0.001
    # model paramater
    DLinear_multi_model = LTSF_DLinear_addsupply(
                                            window_size=window_size,
                                            forcast_size=forcast_size,
                                            kernel_size=25,
                                            individual=True,
                                            feature_size=2,
                                            )
    
    DLinear_model = LTSF_DLinear(
                                    window_size=window_size,
                                    forcast_size=forcast_size,
                                    kernel_size=25,
                                    individual=True,
                                    feature_size=2,
                                    )



    # Loss
    criterion_multi = torch.nn.MSELoss()
    criterion = torch.nn.MSELoss()
    # optimizers
    optimizer_multi = torch.optim.Adam(DLinear_multi_model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(DLinear_multi_model.parameters(), lr=lr)
    
    # # early stopping
    # early_stopping_epochs = 5
    # best_loss = 99999
    # early_stop_counter = 0

    print("strat training")
    print()


    for epoch in tqdm(range(1, epochs+1)):
        loss_multi_list = []
        loss_list = []

        DLinear_multi_model.train()
        DLinear_model.train()

        for batch_eeeeidx, (data, target) in enumerate(train_dl):
            
            optimizer_multi.zero_grad()
            optimizer.zero_grad()

            output_multi = DLinear_multi_model(data)
            output = DLinear_model(data)

            loss_multi = criterion_multi(output_multi, target[:,:,1].unsqueeze(-1))
            loss_multi.backward()

            loss = criterion(output, target[:,:,1].unsqueeze(-1))
            loss.backward()

            optimizer_multi.step()
            optimizer.step()

            loss_multi_list.append(np.sqrt(loss_multi.item()))
            loss_list.append(np.sqrt(loss.item()))


        train_loss_multi_list.append(np.sqrt(np.mean(loss_list)))
        train_loss_list.append(np.sqrt(np.mean(loss_list)))

        DLinear_multi_model.eval()
        DLinear_model.eval()
        with torch.no_grad():
            for data, target in valid_dl:
                output_multi = DLinear_multi_model(data)
                output = DLinear_model(data)


                valid_loss_multi = np.sqrt(criterion_multi(output_multi, target[:,:,1].unsqueeze(-1)).item())
                valid_loss_multi_list.append(np.sqrt(valid_loss_multi.item()))
                valid_loss = np.sqrt(criterion_multi(output, target[:,:,1].unsqueeze(-1)).item())
                valid_loss_list.append(np.sqrt(valid_loss.item()))

        # if valid_loss > best_loss:
        #     early_stop_counter += 1
        # else:
        #     best_loss = valid_loss
        #     early_stop_counter = 0

        # # 조기 종료 조건 확인
        # if early_stop_counter >= early_stopping_epochs:
        #     print(f"Early Stopping epoch:{epoch}")
        #     break
            
        # if epoch == epochs:
        #     data = train_y[-window_size:,].reshape(1,-1)
        #     pred_value = np.array([])
        #     for i in range(28):
        #         pred_input = torch.Tensor(data).reshape(1,window_size,1)
        #         pred_output = DLinear_multi_model(pred_input)

        #         data = np.append(data,pred_output.item())
        #         pred_value = np.append(pred_value,pred_output.item())
        #         data = data[1:]

        #     pred[dataset_code] = pred_value
            # # rescaling pred
            # pred[dataset_code] = min_max_inverse_scaling(pred_value, min_val, max_val)
        
        # print(pred_value)

        # for data, target in test_dl:
        #     output = DLinear_multi_model(data)
        #     test_loss = criterion_multi(output, target.unsqueeze(-1))
        #     test_loss_list.append(test_loss)

        # if valid_loss < max_loss:
        #     torch.save(DLinear_multi_model, f'DLinear_multi_model_{dataset_code}.pth')
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
# weights_list['trend'] = DLinear_multi_model.Linear_Trend.weight.detach().numpy()
# weights_list['seasonal'] = DLinear_multi_model.Linear_Seasonal.weight.detach().numpy()

# for name, w in weights_list.items():    
#     fig, ax = plt.subplots()    
#     plt.title(name)
#     im = ax.imshow(w, cmap='plasma_r',)
#     fig.colorbar(im, pad=0.03)
#     plt.show()
#     plt.savefig("ex.png")



# ## Storing result

# date_col = []
# date = 20230304
# for i in range(28):
#     date_col.append(date + i)
# # print(date_col)



# result = []
# for i in range(len(unique_code)):
#     pred_adj = np.round(pred[unique_code[i]],1)

#     for j in range(28):
#         code = unique_code[i]
#         final = pred_adj[j]
#         if date_col[j] in [20230305, 20230312, 20230319, 20230326]:
#             final = 0
#         else:
#             final = final

#         code_time = f"{unique_code[i]}_{date_col[j]}"
#         result.append([code_time, final])
    
    
# result_df = pd.DataFrame(result, columns = ['ID', 'pred'])
# submission = pd.read_csv("data/sample_submission.csv")


# # 'submission' 데이터프레임과 'result_df' 데이터프레임을 'ID'를 기준으로 병합
# final_submission = submission.merge(result_df, on='ID', how='left')

# # 'pred' 값을 'answer' 열에 복사
# final_submission['answer'] = final_submission['pred']

# # 'pred' 열 삭제
# final_submission.drop('pred', axis=1, inplace=True)

# # 결과 데이터프레임 출력
# # print(final_submission.head())


# # 업데이트된 데이터프레임을 새로운 CSV 파일로 저장
# final_submission.to_csv(f'csv/DLinear_{window_size}_{forcast_size}_kernel3_minmax_vanila.csv', index=False)