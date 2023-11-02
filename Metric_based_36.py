import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from train_baseModel_36 import *
import numpy as np

def RBF(kernel1, kernel2, gamma):
    n1 = kernel1.shape[0]
    n2 = kernel2.shape[0]
    sum = 0.0
    for i in range(n1):
        for j in range(n2):
            # print((torch.mean(kernel1[i, :, :], dim=0)).shape)
            sum += torch.exp(-torch.norm((torch.mean(kernel1[i, :, :], dim=0)-torch.mean(kernel2[j, :, :], dim=0)), p=2)/(2*gamma*gamma))
    sum = sum/(n1*n2)

    return sum
def mmd_loss(source_kernel, target_kernel, gamma= 0.2):
    mmd = 0.0
    mmd += RBF(source_kernel, source_kernel, gamma)
    mmd -= 2*RBF(source_kernel, target_kernel, gamma)
    mmd += RBF(target_kernel, target_kernel, gamma)
    return mmd


target_file = 'CS2_35'
source_file = "CS2_37"
feature_type = ['cc_t', 'cc_cap', 'cc_e', 'slope_cccv_ct', 'start_of_charge_v', 'dis_cap']
dataset_type = '.csv'
if __name__ == '__main__':
    input_dim = 5
    hidden_dim = 6
    out_dim = 1
    step_time = 25

    name = target_file
    path = []
    for i in range(len(feature_type)):
        path.append('D:\\code\\Python\\EE5003\\data\\NASA\\' + name + "\\" + name + "_" + feature_type[i] + dataset_type)
    line = np.array(pd.read_csv(path[0], encoding='utf-8', header=None))
    line = line[1:, :]
    large = np.max(line[:, 0])
    little = np.min(line[:, 0])
    line = (line[:, 0] - little) / (large - little)
    target_data = line.reshape((-1, 1))

    for i in range(1, len(path)):
        line = np.array(pd.read_csv(path[i], encoding='utf-8', header=None))
        line = line[1:, :]
        if i == len(path)-1:
            line = line/1.1
            print(" ")
        else:
            large = np.max(line[:, 0])
            little = np.min(line[:, 0])
            line = (line[:, 0] - little) / (large - little)

        target_data = np.concatenate((target_data, line.reshape((-1, 1))), axis=1)
    # print(target_data[:, 2])
    train_nums = int(len(target_data) * 0.5)
    valid_nums = int(len(target_data) * 0.5)


    target_train_set = []
    for i in range(train_nums - step_time):
        x_ = target_data[i: i + step_time, :-1]
        y_ = target_data[i: i + step_time, -1]
        target_train_set.append((x_, y_))


    target_validation_set = []
    for i in range(train_nums - step_time, valid_nums - step_time):
        x_ = target_data[i: i + step_time, :-1]
        y_ = target_data[i: i + step_time, -1]
        target_validation_set.append((x_, y_))

    target_test_set = []
    for i in range(train_nums - step_time, len(target_data) - step_time):
        x_ = target_data[i: i + step_time, :-1]
        y_ = target_data[i: i + step_time, -1]
        target_test_set.append((x_, y_))
# TARGET DOMAIN
# ------------------------------------------------------------------------------------------------------------
# SOURCE DOMAIN
    name = source_file
    path = []
    for i in range(len(feature_type)):
        path.append('D:\\code\\Python\\EE5003\\data\\NASA\\' + name + "\\" + name + "_"+ feature_type[i] + dataset_type)
    line = np.array(pd.read_csv(path[0], encoding='utf-8', header=None))
    line = line[1:, :]
    large = np.max(line[:, 0])
    little = np.min(line[:, 0])
    line = (line[:, 0] - little) / (large - little)
    source_data = line.reshape((-1, 1))

    for i in range(1, len(path)):
        line = np.array(pd.read_csv(path[i], encoding='utf-8', header=None))
        line = line[1:, :]
        if i == len(path)-1:
            line = line/1.1
            print(" ")
        else:
            large = np.max(line[:, 0])
            little = np.min(line[:, 0])
            line = (line[:, 0] - little) / (large - little)

        source_data = np.concatenate((source_data, line.reshape((-1, 1))), axis=1)

    train_nums = int(len(source_data) * 0.7)
    transfer_nums = int(len(source_data) * 0.5)


    source_train_set = []
    for i in range(train_nums - step_time):
        x_ = source_data[i: i + step_time, :-1]
        y_ = source_data[i: i + step_time, -1]
        source_train_set.append((x_, y_))


    source_transfer_set = []
    for i in range(transfer_nums - step_time):
        x_ = source_data[i: i + step_time, :-1]
        y_ = source_data[i: i + step_time, -1]
        source_transfer_set.append((x_, y_))

    source_test_set = []
    for i in range(train_nums - step_time, len(source_data) - step_time):
        x_ = source_data[i: i + step_time, :-1]
        y_ = source_data[i: i + step_time, -1]
        source_test_set.append((x_, y_))

    print("data is ready")
    model = torch.load('models/base_model_36.pth').cuda()


    for name, param in model.named_parameters():
        param.requires_grad = False

    for param in model.Lstm.parameters():
        param.requires_grad = True
    # for name, param in model.named_parameters():
    #     print(name, param)

    regression_loss = nn.MSELoss()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05)
    model.train()
    # for param_group in optim.param_groups:
    #     print("参数组:")
    #     for param in param_group['params']:
    #         if param.requires_grad:
    #             print(param)

    for epoch in range(10):
        source_kernel = []
        source_kernel = np.array(source_kernel)
        source_kernel = torch.tensor(source_kernel).cuda()

        r_loss = 0
        for train_, label_ in source_train_set:
            train_ = torch.tensor(train_).float().reshape((-1, step_time, input_dim)).cuda()
            label_ = torch.tensor(label_).float().reshape((-1, step_time, out_dim)).cuda()

            source_out, _ = model(train_)
            # source_kernel = torch.cat((source_kernel, source_feature))
            r_loss += regression_loss(source_out, label_)
        # print(source_kernel.shape)

        for x_, _ in source_transfer_set:
            x_ = torch.tensor(x_).float().reshape((-1, step_time, input_dim)).cuda()
            _, source_feature = model(x_)
            source_kernel = torch.cat((source_kernel, source_feature))

        target_kernel = []
        target_kernel = np.array(target_kernel)
        target_kernel = torch.tensor(target_kernel).cuda()
        for x_, _ in target_train_set:
            x_ = torch.tensor(x_).float().reshape((-1, step_time, input_dim)).cuda()
            target_out, target_feature = model(x_)
            target_kernel = torch.cat((target_kernel, target_feature))

        print(target_kernel.shape, source_kernel.shape)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # source_points = np.zeros((1, 3))
        # target_points = np.zeros((1, 3))
        # for i in range(source_kernel.shape[0]):
        #     for j in range(source_kernel.shape[1]):
        #         point = source_kernel[i, j, :]
        #         point = np.array(point.cpu().detach()).reshape((-1 ,3))
        #         # print(point.shape)
        #         source_points = np.vstack((source_points, point))
        #         # plt.plot(point[0], point[1], point[2], 'red')
        # for i in range(target_kernel.shape[0]):
        #     for j in range(target_kernel.shape[1]):
        #         point = target_kernel[i, j, :]
        #         point = np.array(point.cpu().detach()).reshape((-1, 3))
        #         target_points = np.vstack((target_points, point))
        #         # plt.plot(point[0], point[1], point[2], 'blue')
        # plt.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='red')
        # plt.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='blue')
        # print(target_points.shape)
        # plt.show()

        r_loss = torch.mean(r_loss)
        h_lambda = 0.7
        t_loss = mmd_loss(source_kernel, target_kernel)

        total_loss = t_loss*h_lambda+ r_loss

        print(r_loss)
        print(t_loss)
        # if r_loss<0.01 and t_loss<0.01:
        #     break
        model.zero_grad()
        total_loss.backward()
        optim.step()

        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(name, param)
                print(param.grad)

    model.eval()
    pred_list = []
    real_list = []
    for (x_test, label_test) in target_test_set:
        x_test = torch.tensor(x_test).float().reshape((-1, step_time, input_dim)).cuda()
        label_test = torch.tensor(label_test).float().reshape((-1, step_time, out_dim)).cuda()
        pred, _ = model(x_test)
        pred_list.append(pred[:, -1, :].item())
        real_list.append(label_test[:, -1, :].item())
    # print(len(pred_list))
    # print(len(real_list))

    residuals = np.array(pred_list) - np.array(real_list)
    rmse = np.sqrt(np.mean(residuals ** 2))
    print("rmse: {}".format(rmse.item()))
    plt.plot(real_list)
    plt.plot(pred_list)
    plt.legend(("real", "pred"))
    plt.show()


