import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(     # input_size=[128, 1, 32, 32]
            nn.Conv2d(1, 6, 5, 1, 2), # in_channels, out_channels, kernel_size, stride, padding，padding=2保证输入输出尺寸相同
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=[128, 6, 16, 16]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # output_size=[128, 16, 6, 6]
        )
        self.fc1 = nn.Sequential(
            nn.Linear(576, 576),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(576, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 1)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = Variable(torch.squeeze(x, dim=0).float(), requires_grad=False)
        # input: [128, 1, 32, 32]，batch size: 128
        x = self.conv1(x) # output: [128, 6, 16, 16]
        x = self.conv2(x) # output: [128, 16, 6, 6]
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size(0), -1) # output: [128, 576], 576 = 16 * 6 * 6
        x = self.fc1(x) # output: [128, 576]
        x = self.fc2(x) # output: [128, 84]
        x = self.fc3(x) # # output: [128, 1]
        return x