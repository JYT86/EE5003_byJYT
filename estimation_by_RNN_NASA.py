import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
class RNN(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.rnn = nn.RNN(input_size=in_dim, hidden_size=hid_dim, num_layers=3, batch_first=True)#x:[batch_size, step_time, in_dim]
        self.linear = nn.Linear(hid_dim, out_dim)
        self.hid_dim = hid_dim
        self.out_dim = out_dim

    def forward(self, x):
        """
        x:[step_time, batch_size, in_dim]
        """
        outputs, hn = self.rnn(x)
        outputs = outputs.reshape(-1, self.hid_dim)
        outputs = self.linear(outputs)
        outputs = outputs.reshape(x.shape[0], -1, self.out_dim)
        assert(outputs.shape == (x.shape[0], x.shape[1], self.out_dim))
        return outputs, hn


battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
end_dis = [2.7, 2.5, 2.2, 2.5]
dataset_type = '.csv'
feature_type = ['_1cc_t', '_1cv_t', '_1dis_cap', '_1cc_cap', '_1cv_cap', '_1cc_e', '_1cv_e', '_1mean_cc_i', '_1a_cvcc_ct', '_1b_cvcc_ct', '_1slope_cccv_ct', '_1start_of_charge_v']
if __name__ == '__main__':

    batch_size = len(battery_list)
    step_time = 100
    in_dim = 12
    hid_dim = 8
    out_dim = 1
    input = np.zeros((batch_size, step_time, in_dim))
    label = np.zeros((batch_size, step_time, out_dim))
    for i in range(len(battery_list)):
        name = battery_list[i]
        cc_t_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[0]+dataset_type
        cv_t_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[1]+dataset_type
        dis_cap_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[2]+dataset_type
        cc_cap_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[3]+dataset_type
        cv_cap_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[4]+dataset_type
        cc_e_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[5]+dataset_type
        cv_e_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[6]+dataset_type
        mean_cc_i_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[7]+dataset_type
        a_cvcc_ct_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[8]+dataset_type
        b_cvcc_ct_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[9]+dataset_type
        slope_cccv_ct_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[10]+dataset_type
        start_of_charge_path = 'D:\\code\\Python\\EE5003\\data\\NASA\\'+name+"\\"+name+feature_type[11]+dataset_type

        cv_duration = pd.read_csv(cc_t_path, encoding='utf-8', header=None)
        cc_duration = pd.read_csv(cv_t_path, encoding='utf-8', header=None)
        dis_cap = pd.read_csv(dis_cap_path, encoding='utf-8', header=None)
        cc_cap = pd.read_csv(cc_cap_path, encoding='utf-8', header=None)
        cv_cap = pd.read_csv(cv_cap_path, encoding='utf-8', header=None)
        cc_e = pd.read_csv(cc_e_path, encoding='utf-8', header=None)
        cv_e = pd.read_csv(cv_e_path, encoding='utf-8', header=None)
        mean_cc_i = pd.read_csv(mean_cc_i_path, encoding='utf-8', header=None)
        a_cvcc_ct = pd.read_csv(a_cvcc_ct_path, encoding='utf-8', header=None)
        b_cvcc_ct = pd.read_csv(b_cvcc_ct_path, encoding='utf-8', header=None)
        slope_cccv_ct = pd.read_csv(slope_cccv_ct_path, encoding='utf-8', header=None)
        start_of_charge = pd.read_csv(start_of_charge_path, encoding='utf-8', header=None)

        cv_duration = np.array(cv_duration)
        cc_duration = np.array(cc_duration)
        dis_cap = np.array(dis_cap)
        cc_cap = np.array(cc_cap)
        cv_cap = np.array(cv_cap)
        cc_e = np.array(cc_e)
        cv_e = np.array(cv_e)
        mean_cc_i = np.array(mean_cc_i)
        a_cvcc_ct = np.array(a_cvcc_ct)
        b_cvcc_ct = np.array(b_cvcc_ct)
        slope_cccv_ct = np.array(slope_cccv_ct)
        start_of_charge = np.array(start_of_charge)

        [r, c] = cv_duration.shape
        print(r, c)

        input[i, :, 0] = cv_duration[0:step_time, 0]
        input[i, :, 1] = cc_duration[0:step_time, 0]
        input[i, :, 2] = cc_e[0:step_time, 0]
        input[i, :, 3] = cv_e[0:step_time, 0]
        input[i, :, 4] = cc_cap[0:step_time, 0]
        input[i, :, 5] = cv_cap[0:step_time, 0]
        input[i, :, 6] = mean_cc_i[0:step_time, 0]
        input[i, :, 7] = a_cvcc_ct[0:step_time, 0]
        input[i, :, 8] = b_cvcc_ct[0:step_time, 0]
        input[i, :, 9] = slope_cccv_ct[0:step_time, 0]
        input[i, :, 10] = start_of_charge[0:step_time, 0]
        input[i, :, 11] += end_dis[i]

        label[i, :, 0] = dis_cap[0:step_time, 0]


    # print(cv_duration)

    lr = 0.01


    print(input.shape)



    #print(input)

    input = torch.Tensor(input).cuda()
    train = input[:-1, :, :]
    test = input[-1, :, :]
    test = test.reshape((1, -1, in_dim))
    label = (torch.Tensor(label)/2).cuda()
    train_label = label[:-1, :, :]
    test_label = label[-1, :, :]
    test_label = test_label.reshape((1, -1, 1))
    print(train.shape)
    print(test.shape)

    ji = RNN(in_dim, hid_dim, out_dim).cuda()
    loss_func = nn.MSELoss()
    optimizier = optim.Adam(ji.parameters(), lr)

    for epoch in range(10000):
        output, hn = ji(train)

        loss = loss_func(output, train_label)

        ji.zero_grad()
        loss.backward()

        # avoid gradient exploding
        max_grad_norm = 10
        clip_grad_norm_(ji.parameters(), max_grad_norm)
        optimizier.step()

        if epoch%1000 ==0:
            # print(output.shape, label.shape)
            print(loss)


    #predict
    pred_out, hn = ji(train[0,:,:].reshape(1,-1,in_dim))
    # print(pred_out)
    print(pred_out.shape)
    pred_out = torch.reshape(pred_out, (-1, 1))
    test_label = torch.reshape(train_label[0,:,:], (-1, 1))

    print(pred_out.shape)
    print(test_label.shape)

    residuals = pred_out - test_label
    rmse = torch.sqrt(torch.mean(residuals ** 2))
    print("rmse: {}".format(rmse.item()))

    plt.plot(pred_out.cpu().detach().numpy())
    plt.legend("real")
    plt.plot(test_label.cpu().numpy())
    plt.legend("prediction")
    plt.xlabel("cycles")
    plt.ylabel("SOH")
    plt.grid("on")
    plt.show()




