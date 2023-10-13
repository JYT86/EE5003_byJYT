import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from train_base_model import Base_model
import matplotlib.pyplot as plt

base_file = "B0005"
feature_type = ['_1cc_t', '_1cv_t',  '_1cc_cap', '_1cv_cap', '_1cc_e', '_1cv_e', '_1mean_cc_i', '_1a_cvcc_ct', '_1b_cvcc_ct', '_1slope_cccv_ct', '_1start_of_charge_v','_1dis_cap']
dataset_type = '.csv'

if __name__ == "__main__":
    name = base_file
    path = []
    for i in range(len(feature_type)):
        path.append('D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[i]+dataset_type)

    line = np.array(pd.read_csv(path[0], encoding='utf-8', header=None))
    line = line[1:, :]
    large = np.max(line[:, 0])
    little = np.min(line[:, 0])
    line = (line[:, 0] - little) / (large - little)
    data = line.reshape((-1, 1))
    # print(data.shape)

    for i in range(1, len(path)):
        line = np.array(pd.read_csv(path[i], encoding='utf-8', header=None))
        line = line[1:, :]
        if i == len(path) - 1:
            line = line / 2
            print(" ")
        else:
            large = np.max(line[:, 0])
            little = np.min(line[:, 0])
            line = (line[:, 0] - little) / (large - little)

        data = np.concatenate((data, line.reshape((-1, 1))), axis=1)
    # print(data[:,-1])

    train_nums = int(len(data) * 0.2)
    valid_nums = int(len(data) * 0.4)

    input_dim = 11
    hidden_dim = 20
    out_dim = 1
    step_time = 5

    train_set = []
    for i in range(train_nums-step_time):
        x_ = data[i: i+step_time, :-1]
        y_ = data[i: i+step_time, -1]
        train_set.append((x_, y_))

    validation_set = []
    for i in range(train_nums - step_time, valid_nums - step_time):
        x_ = data[i: i + step_time, :-1]
        y_ = data[i: i + step_time, -1]
        validation_set.append((x_, y_))

    model = torch.load("models/base_model.pth")
    # print(model)

    for name, param in model.named_parameters():
        param.requires_grad = False
        # print(name, param)
    model.Linear1 = nn.Linear(hidden_dim, out_dim)
    for param in model.Linear1.parameters():
        param.requires_grad = True
    # print("**************************************")

    # for name, param in model.named_parameters():
    #
    #     print(param)

    loss = nn.MSELoss()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05)
    model.train()

    best_accurancy = 100
    best_model = model
    for epoch in range(100):
        for x_train, label_train in train_set:
            x_train = torch.tensor(x_train).float().reshape((-1, step_time, input_dim))
            label_train = torch.tensor(label_train).float().reshape((-1, step_time, out_dim))

            out = model(x_train)
            l = loss(out, label_train)

            model.zero_grad()
            max_grad_norm = 10

            l.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)

            optim.step()

        vl = 0
        for x_valid, label_valid in validation_set:
            x_valid = torch.tensor(x_valid).float().reshape((-1, step_time, input_dim))
            label_valid = torch.tensor(label_valid).float().reshape((-1, step_time, out_dim))

            out_val = model(x_valid)

            l = loss(out_val, label_valid)
            vl += l.item()

        if vl<=best_accurancy:
            best_model = model
            best_accurancy = vl
        print(vl)

    # test
    test_set = []
    for i in range(valid_nums - step_time, len(data) - step_time):
        x_ = data[i: i + step_time, :-1]
        y_ = data[i: i + step_time, -1]
        test_set.append((x_, y_))

    model.eval()
    pred_list = []
    real_list = []
    for x_test, label_test in test_set:
        x_test = torch.tensor(x_test).float().reshape((-1, step_time, input_dim))
        label_test = torch.tensor(label_test).float().reshape((-1, step_time, out_dim))
        out_test = best_model(x_test)
        pred_list.append(out_test[:, -1, :].item())
        real_list.append(label_test[:, -1, :].item())

    residuals = np.array(pred_list) - np.array(real_list)
    rmse = np.sqrt(np.mean(residuals ** 2))
    print("rmse: {}".format(rmse.item()))
    plt.plot(real_list)
    plt.plot(pred_list)
    plt.legend(("real", "pred"))
    plt.show()




