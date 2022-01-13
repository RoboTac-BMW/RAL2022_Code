import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


def coral(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)
    return c


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

class get_coral_loss(torch.nn.Module):
    def __init__(self, DA_alpha=0.1, DA_lamda=0.5, mat_diff_loss_scale=0.001):
        super(get_coral_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.DA_lamda = DA_lamda
        self.DA_alpha = DA_alpha

    def forward(self, pred, target, trans_feat, feature_dense, feature_sparse):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        coral_loss = coral(feature_dense, feature_sparse)
        total_loss = self.DA_alpha * loss + mat_diff_loss * self.mat_diff_loss_scale + self.DA_lamda * coral_loss
        return total_loss


class get_mmd_loss(torch.nn.Module):
    def __init__(self, DA_alpha=0.1, DA_lamda=0.5,
                 mat_diff_loss_scale=0.001, kernel_mul = 2.0, kernel_num = 5):
        super(get_mmd_loss, self).__init__()
        self.DA_lamda = DA_lamda
        self.DA_alpha = DA_alpha
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)),
                                           int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)),
                                           int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, pred, target, trans_feat, feature_dense, feature_sparse):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        batch_size = int(pred.size()[0])
        kernels = self.guassian_kernel(feature_dense, feature_sparse, kernel_mul=self.kernel_mul,
                                       kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        mmd_loss = torch.mean(XX + YY - XY -YX)
        total_loss =self.DA_alpha * loss + mat_diff_loss * self.mat_diff_loss_scale + self.DA_lamda * mmd_loss
        return total_loss

class get_coral_mmd_loss(get_mmd_loss):
    def __init__(self, DA_alpha=0.1, DA_lamda=0.5,
                 mat_diff_loss_scale=0.001, kernel_mul = 2.0, kernel_num = 5):
        super(get_coral_mmd_loss, self).__init__(DA_alpha=0.1, DA_lamda=0.5,
                                                 mat_diff_loss_scale=0.001,
                                                 kernel_mul = 2.0, kernel_num = 5)

    def forward(self, pred, target, trans_feat, feature_dense, feature_sparse):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        batch_size = int(pred.size()[0])
        kernels = self.guassian_kernel(feature_dense, feature_sparse, kernel_mul=self.kernel_mul,
                                       kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        mmd_loss = torch.mean(XX + YY - XY -YX)
        coral_loss = coral(feature_dense, feature_sparse)
        # total_loss =self.DA_alpha * loss + mat_diff_loss * self.mat_diff_loss_scale +
        #             self.DA_lamda * mmd_loss
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale + coral_loss + mmd_loss
        return total_loss









