import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
import matplotlib.pyplot as plt
import tqdm
from torch.nn.utils import clip_grad_norm_
class Base_model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True, num_layers=1)
        self.Linear = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        out, _ = self.Lstm(x)  #x.shape = (batch_size, seq_len, in_dim)
        out = self.Linear(out) #out.shape = (batch_size, seq_len, hid_dim)
        assert(out.shape == (x.shape[0], x.shape[1], self.out_dim))
        return out

# data load and processing
base_file = "B0007"
feature_type = ['_1cc_t', '_1cv_t',  '_1cc_cap', '_1cv_cap', '_1cc_e', '_1cv_e', '_1mean_cc_i', '_1a_cvcc_ct', '_1b_cvcc_ct', '_1slope_cccv_ct', '_1start_of_charge_v','_1dis_cap']
dataset_type = '.csv'
if __name__ == '__main__':
    name = base_file
    path = []
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[0]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[1]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[2]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[3]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[4]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[5]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[6]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[7]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[8]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[9]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[10]+dataset_type)
    path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[11]+dataset_type)
    line = np.array(pd.read_csv(path[0], encoding='utf-8', header=None))
    line = line[1:, :]
    large = np.max(line[:, 0])
    little = np.min(line[:, 0])
    line = (line[:, 0]-little)/(large - little)
    data = line.reshape((-1, 1))
    print(data.shape)

    for i in range(1, len(path)):
        line = np.array(pd.read_csv(path[i], encoding='utf-8', header=None))
        line = line[1:, :]
        if i == len(path)-1:
            line = line/2
            print(" ")
        else:
            large = np.max(line[:, 0])
            little = np.min(line[:, 0])
            line = (line[:, 0] - little) / (large - little)

        data = np.concatenate((data, line.reshape((-1, 1))), axis=1)



    print(data[:,-1])

    train_nums = int (len(data)*0.4)
    valid_nums = int(len(data)*0.6)
    print(train_nums)

    input_dim = 11
    hidden_dim = 20
    out_dim = 1
    step_time = 5

    train_set = []
    for i in range(train_nums-step_time):
        x_ = data[i: i+step_time, :-1]
        y_ = data[i: i+step_time, -1]
        train_set.append((x_, y_))
    print(train_set)

    validation_set = []
    for i in range(train_nums-step_time, valid_nums-step_time):
        x_ = data[i: i + step_time, :-1]
        y_ = data[i: i + step_time, -1]
        validation_set.append((x_, y_))

    model = Base_model(in_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    loss = nn.MSELoss()
    optim = optim.Adam(model.parameters(), lr=0.05)

    model.train()
    best_model = model
    best_accurancy = 1000000
    for epoch in range(100):
        for (train_, label_ ) in validation_set:
            train_ = torch.tensor(train_).float().reshape((-1, step_time, input_dim))
            label_ = torch.tensor(label_).float().reshape((-1, step_time, out_dim))

            # print(label_.shape)

            out = model(train_)
            l = loss(out, label_)

            model.zero_grad()
            max_grad_norm = 10

            l.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)

            optim.step()

        #validation
        vl = 0
        for (valid_, label_) in validation_set:
            valid_ = torch.tensor(valid_).float().reshape((-1, step_time, input_dim))
            label_ = torch.tensor(label_).float().reshape((-1, step_time, out_dim))
            out = model(valid_)
            vl += loss(out, label_).item()
        if(vl<=best_accurancy):
            best_accurancy = vl
            best_model = model
        print(vl)

    #predict
    model.eval()
    test_set = []
    for i in range(valid_nums-step_time, len(data)-step_time):
        x_ = data[i: i + step_time, :-1]
        y_ = data[i: i + step_time, -1]
        test_set.append((x_, y_))
    print(len(test_set))

    pred_list = []
    real_list = []
    for (x_test, label_test) in test_set:
        x_test = torch.tensor(x_test).float().reshape((-1, step_time, input_dim))
        label_test = torch.tensor(label_test).float().reshape((-1, step_time, out_dim))
        pred = best_model(x_test)
        pred_list.append(pred[:, -1, :].item())
        real_list.append(label_test[:, -1, :].item())
    print(len(pred_list))
    print(len(real_list))
    plt.plot(real_list)
    plt.plot(pred_list)
    plt.legend(("real","pred"))
    plt.show()
