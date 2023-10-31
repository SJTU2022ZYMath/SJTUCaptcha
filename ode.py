"""
用神经网络模拟微分方程,f(x)'=f(x),初始条件f(0) = 1
"""
import torch
# import numpy
# # from torch.autograd import Variable
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self, layer_num, nerve_num):
        # layer_number是有多少隐藏层
        # nerve_number是每层的神经元数量
        super(Net, self).__init__()
        self.input_layer = torch.nn.Linear(1, nerve_num)
        self.hidden_layer = torch.nn.ModuleList([torch.nn.Linear(nerve_num, nerve_num) for i in range(layer_num)])
        self.output_layer = torch.nn.Linear(nerve_num, 1)

    def forward(self, x):
        o = self.activation_fn(self.input_layer(x))
        for i, li in enumerate(self.hidden_layer):
            o = self.activation_fn(li(o))
        out = self.output_layer(o)
        return out

    # 激活函数
    def activation_fn(self, x):
        return torch.tanh(x)

# 主程序
if __name__ == "__main__":
    plot_args={"dash_capstyle":"round", "dash_joinstyle":"round", "solid_capstyle":"round", "solid_joinstyle":"round"}

    x = torch.linspace(-2, 2, 2000, requires_grad=True).unsqueeze(-1)
    y = torch.exp(x)
    net = Net(4, 18)
    lr = 1e-4
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(),lr)
    plt.ion()
    for i in range(1,8001):

        y_0 = net(torch.zeros(1))
        dx = torch.autograd.grad(net(x), x, grad_outputs=torch.ones_like(net(x)), create_graph=True)[0]
        optimizer.zero_grad()
        y_train = net(x)
        Mse1 = loss_fn(y_train,dx)
        Mse2 = loss_fn(y_0,torch.ones(1))
        loss = Mse1 + Mse2
        if i % 100 == 0:
            plt.cla()
            plt.scatter(x.detach().numpy(),y.detach().numpy())
            plt.plot(x.detach().numpy(), y_train.detach().numpy(), c='red', label='估计值', lw=4,**plot_args)
            plt.text(-2, 6, f'Loss={loss.item():.5f}', fontdict={'size':18, 'color': 'red'})
            plt.pause(0.1)
            print(f'第{i}轮 - lr：{lr} - 损失：{loss.item()} - y(0)：{y_0}')
        loss.backward()
        optimizer.step()
    plt.ioff()
    plt.show()
    y_1 = net(torch.ones(1))
    print(f'y_1:{y_1}')
    y_train_final = net(x)
    plt.plot(x.detach().numpy(), y.detach().numpy(), c='blue', label='真实值', lw=6, **plot_args)
    plt.plot(x.detach().numpy(), y_train_final.detach().numpy(), c='red', label='估计值', lw=4, **plot_args)
    plt.legend(loc='best')
    plt.show()