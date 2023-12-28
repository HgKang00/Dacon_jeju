import torch

class moving_avg(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual 
    
# class LTSF_DLinear_addsupply(torch.nn.Module):
#     def __init__(self, window_size, forcast_size, kernel_size, individual, feature_size):
#         super(LTSF_DLinear_addsupply, self).__init__()
#         self.window_size = window_size
#         self.forcast_size = forcast_size
#         self.decompsition = series_decomp(kernel_size)
#         self.individual = individual
#         self.channels = feature_size
#         if self.individual:
#             self.Linear_Seasonal = torch.nn.ModuleList()
#             self.Linear_Trend = torch.nn.ModuleList()
#             for i in range(self.channels):
#                 self.Linear_Trend.append(torch.nn.Linear(self.window_size, self.forcast_size))
#                 self.Linear_Trend[i].weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))
#                 self.Linear_Seasonal.append(torch.nn.Linear(self.window_size, self.forcast_size))
#                 self.Linear_Seasonal[i].weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))
#                 self.Linear_Seasonal.append(torch.nn.Linear(self.forcast_size, self.forcast_size))
#         else:
#             self.Linear_Trend = torch.nn.Linear(self.window_size, self.forcast_size)
#             self.Linear_Trend.weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))
#             self.Linear_Seasonal = torch.nn.Linear(self.window_size,  self.forcast_size)
#             self.Linear_Seasonal.weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))

#         self.Linear_Supply = torch.nn.Linear(self.forcast_size, self.forcast_size)
#         self.Linear_Supply.weight = torch.nn.Parameter((1/100)*torch.ones([self.forcast_size, self.forcast_size]))

            


#     def forward(self, x):
#         trend_init, seasonal_init = self.decompsition(x)
#         trend_init, seasonal_init = trend_init.permute(0,2,1), seasonal_init.permute(0,2,1)
#         if self.individual:
#             trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.forcast_size], dtype=trend_init.dtype).to(trend_init.device)
#             seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.forcast_size], dtype=seasonal_init.dtype).to(seasonal_init.device)
#             for idx in range(self.channels):
#                 trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
#                 seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])                
#         else:
#             trend_output = self.Linear_Trend(trend_init)
#             seasonal_output = self.Linear_Seasonal(seasonal_init)


#         x = seasonal_output[:,0,:] + trend_output[:,0,:] + seasonal_output[:,1,:] + trend_output[:,1,:]
        
        
#         return x
    

class LTSF_DLinear_addsupply(torch.nn.Module):
    def __init__(self, window_size, forcast_size, kernel_size, hidden_size):
        super(LTSF_DLinear_addsupply, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.decomposition = series_decomp(kernel_size)
        self.hidden_size = hidden_size


        self.NonLinear_Trend = torch.nn.Sequential(
                torch.nn.Linear(self.window_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.forcast_size)
            )
        self.NonLinear_Seasonal = torch.nn.Sequential(
                torch.nn.Linear(self.window_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.forcast_size)
            )
    
        # Linear layers for the second component (index 1)
        self.Linear_Trend= torch.nn.Linear(self.window_size, self.forcast_size)
        self.Linear_Seasonal = torch.nn.Linear(self.window_size, self.forcast_size)

    def forward(self, x):
        # Decomposition
        trend_init, seasonal_init = self.decomposition(x)
        trend_init, seasonal_init = trend_init.permute(0, 2, 1), seasonal_init.permute(0, 2, 1)

        # Apply non-linear MLP for index 0
        trend_output_0 = self.NonLinear_Trend(trend_init[:, 0, :])
        seasonal_output_0 = self.NonLinear_Seasonal(trend_init[:, 0, :])

        trend_output_1 = self.Linear_Trend(trend_init[:, 1, :])
        seasonal_output_1 = self.Linear_Seasonal(seasonal_init[:, 1, :])

        # Combine the results
        x = seasonal_output_0 + trend_output_0 + seasonal_output_1 + trend_output_1
        
        return x
