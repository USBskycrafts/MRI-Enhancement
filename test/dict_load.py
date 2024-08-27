from collections import OrderedDict
import torch
import torch.nn as nn


class ChildNetwork(nn.Module):
    def __init__(self):
        super(ChildNetwork, self).__init__()
        self.linear = nn.Linear(10, 5)  # 假设的子网络层

    def forward(self, x):
        return self.linear(x)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        print("Loading state dict in ParentNetwork")
        print(state_dict)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class ParentNetwork(nn.Module):
    def __init__(self):
        super(ParentNetwork, self).__init__()
        self.child = ChildNetwork()  # 子网络作为主网络的一部分
        self.linear = nn.Linear(5, 2)  # 主网络的其他层

    def forward(self, x):
        x = self.child(x)  # 先通过子网络
        x = torch.relu(x)
        return self.linear(x)  # 然后通过主网络的层


# 创建模型实例
model = ParentNetwork()

# 模拟一个状态字典
state_dict = OrderedDict({
    'child.linear.weight': torch.randn(5, 10),
    'child.linear.bias': torch.randn(5),
    'linear.weight': torch.randn(2, 5),
    'linear.bias': torch.randn(2)
})

# 加载状态字典
model.load_state_dict(state_dict)

# 验证模型参数是否加载
